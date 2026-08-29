#!/usr/bin/env python
"""Fit a pairwise XGBoost ranker on rewards from multiple LoRA/value heads."""

import glob
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xgboost as xgb
from accelerate import Accelerator, InitProcessGroupKwargs
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from grm_utils import AutoModelForCausalLMWithMultiValueHead


@dataclass
class ScriptArguments:
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit folder containing adapter_0, adapter_1, ..."},
    )
    adapter_glob: str = field(
        default="./reward_models_train/**/best_step_*_acc*/",
        metadata={"help": "Checkpoint path or fallback glob"},
    )
    base_model: str = field(default="google/gemma-2b-it")
    max_length: int = field(default=1024)
    batch_size: int = field(default=8)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    attn_implementation: str = field(default="sdpa")

    tree_method: str = field(default="gpu_hist")
    num_round: int = field(default=256)
    early_stopping: int = field(default=20)
    booster_out: str = field(default="multi_lora_booster.xgb")
    xgb_max_depth: int = field(default=5)
    xgb_eta: float = field(default=0.05)
    xgb_min_child_weight: float = field(default=30.0)
    xgb_gamma: float = field(default=2.0)
    xgb_reg_lambda: float = field(default=0.1)
    distributed_timeout_minutes: int = field(default=1200)

    layer_type: Optional[str] = field(default="mlp")
    num_layers: int = field(default=1)
    num_neurons: int = field(default=1024)
    dataset: str = field(default="llm-blender/Unified-Feedback")
    eval_dataset: str = field(default="llm-blender/Unified-Feedback")
    dataset_mode: str = field(default="40K-heldout")
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Optional extraction cap for smoke tests"}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Optional validation cap for smoke tests"}
    )


@dataclass(frozen=True)
class AdapterCheckpoint:
    index: int
    name: str
    outer_dir: Path
    weights_dir: Path
    value_head_path: Path


class PairAccCB(xgb.callback.TrainingCallback):
    """Evaluate, save and early-stop on pair accuracy in one ordered callback."""

    def __init__(self, dvalid, checkpoint_path, early_stopping):
        self.dvalid = dvalid
        self.group_ptr = dvalid.get_uint_info("group_ptr")
        if len(self.group_ptr) < 2 or not np.all(np.diff(self.group_ptr) == 2):
            raise ValueError("Pair accuracy requires every XGBoost group to have size 2")
        self.curr_acc = 0.0
        self.best_acc = -1.0
        self.best_iteration = -1
        self.checkpoint_path = Path(checkpoint_path)
        self.early_stopping = early_stopping

    def after_iteration(self, model, epoch, evals_log):
        pred = model.predict(self.dvalid)
        starts = self.group_ptr[:-1]
        self.curr_acc = float(np.mean(pred[starts] > pred[starts + 1]))
        print(f"[{epoch}] val-pair_acc: {self.curr_acc:.4f}")
        if self.curr_acc > self.best_acc + 1e-7:
            self.best_acc = self.curr_acc
            self.best_iteration = epoch
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.set_attr(
                feature_mode="response",
                best_iteration=str(epoch),
                best_pair_accuracy=f"{self.curr_acc:.8f}",
            )
            model.save_model(str(self.checkpoint_path))
            print(
                f"[{epoch}] new best pair accuracy {self.curr_acc:.4f}; "
                f"saved to {self.checkpoint_path}"
            )
        return (
            self.early_stopping > 0
            and epoch - self.best_iteration >= self.early_stopping
        )


def resolve_checkpoint_root(
    checkpoint_dir: Optional[str], adapter_glob: str
) -> Path:
    """Resolve an explicit checkpoint directory or the first glob match."""
    candidate = checkpoint_dir or adapter_glob
    direct = Path(candidate).expanduser()
    if direct.is_dir():
        return direct.resolve()

    matches = sorted(
        Path(path).expanduser().resolve()
        for path in glob.glob(candidate, recursive=True)
        if Path(path).is_dir()
    )
    if not matches:
        raise FileNotFoundError(f"No checkpoint directory matches {candidate!r}")
    if len(matches) > 1:
        print(f"Found {len(matches)} checkpoints; using {matches[0]}")
    return matches[0]


def discover_adapter_checkpoints(root: Path) -> list[AdapterCheckpoint]:
    """Discover nested and legacy-flat LoRA/value-head checkpoint layouts."""
    adapters = []
    for outer in root.glob("adapter_*"):
        if not outer.is_dir():
            continue
        suffix = outer.name.removeprefix("adapter_")
        if not suffix.isdigit():
            continue

        index = int(suffix)
        name = f"adapter_{index}"
        weights_dir = next(
            (
                path
                for path in (outer / name, outer)
                if (path / "adapter_config.json").is_file()
                and (path / "adapter_model.safetensors").is_file()
            ),
            None,
        )
        if weights_dir is None:
            raise FileNotFoundError(
                f"{outer} has no complete LoRA payload. Expected both "
                "adapter_config.json and adapter_model.safetensors."
            )

        value_head_path = next(
            (
                path
                for path in (outer / "v_head.bin", weights_dir / "v_head.bin")
                if path.is_file()
            ),
            None,
        )
        if value_head_path is None:
            raise FileNotFoundError(f"No v_head.bin found for {name} in {outer}")

        adapters.append(
            AdapterCheckpoint(
                index=index,
                name=name,
                outer_dir=outer,
                weights_dir=weights_dir,
                value_head_path=value_head_path,
            )
        )

    adapters.sort(key=lambda item: item.index)
    if not adapters:
        raise RuntimeError(f"No adapter_N directories found in {root}")

    indices = [item.index for item in adapters]
    expected = list(range(len(adapters)))
    if indices != expected:
        raise ValueError(
            f"Adapters must be contiguous and zero-based; found {indices}, "
            f"expected {expected}"
        )
    return adapters


def _fix_lora_keys(state, dtype):
    """Map legacy wrapper prefixes to the current multi-value-head wrapper."""
    fixed = {}
    for key, value in state.items():
        if ".pretrained_model." not in key:
            key = key.replace(
                "base_model.model.model.",
                "base_model.model.pretrained_model.model.",
                1,
            )
        fixed[key] = value.to(dtype)
    return fixed


def load_adapters_and_heads(
    base_model, adapters: list[AdapterCheckpoint], device, dtype
):
    """Load all LoRA adapters and their matching value heads."""
    first = adapters[0]
    first_config = LoraConfig.from_pretrained(first.weights_dir)
    model = get_peft_model(
        base_model,
        first_config,
        adapter_name=first.name,
        mixed=False,
    )

    for adapter in adapters:
        if adapter.name not in model.peft_config:
            config = LoraConfig.from_pretrained(adapter.weights_dir)
            model.add_adapter(adapter.name, config)

        lora_state = load_file(
            adapter.weights_dir / "adapter_model.safetensors", device="cpu"
        )
        set_peft_model_state_dict(
            model,
            _fix_lora_keys(lora_state, dtype),
            adapter_name=adapter.name,
        )

        head_state = torch.load(
            adapter.value_head_path, map_location="cpu", weights_only=True
        )
        model.v_heads[adapter.index].load_state_dict(head_state, strict=True)
        print(
            f"Loaded {adapter.name}: {len(lora_state)} LoRA tensors and "
            f"{len(head_state)} value-head tensors"
        )

    model.set_adapter(first.name)
    return model.to(device)


def load_checkpoint_tokenizer(checkpoint_root: Path, base_model: str, max_length: int):
    """Prefer the training tokenizer and fail clearly if no chat template exists."""
    errors = []
    tokenizer = None
    for source in (str(checkpoint_root), base_model):
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
            print(f"Loaded tokenizer from {source}")
            break
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    if tokenizer is None:
        raise RuntimeError("Could not load tokenizer:\n" + "\n".join(errors))

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer has no chat template. Save the tokenizer used for reward-model "
            "training in the checkpoint instead of guessing a model-specific template."
        )

    tokenizer.max_length = max_length
    tokenizer.model_max_length = max_length
    return tokenizer


def _to_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _truncate(values, max_length, side):
    if len(values) <= max_length:
        return values
    if side == "left":
        return values[-max_length:]
    return values[:max_length]


def _pair_features(tokenizer, batch, side, max_length):
    features = []
    id_key = f"input_ids_{side}"
    mask_key = f"attention_mask_{side}"
    truncation_side = getattr(tokenizer, "truncation_side", "right")

    for row_number, example in enumerate(batch):
        input_ids = _to_list(example[id_key])
        attention_mask = _to_list(example[mask_key])
        if len(input_ids) != len(attention_mask):
            raise ValueError(
                f"{side} row {row_number}: input length {len(input_ids)} != "
                f"attention-mask length {len(attention_mask)}"
            )

        input_ids = _truncate(input_ids, max_length, truncation_side)
        attention_mask = _truncate(attention_mask, max_length, truncation_side)
        attention_mask = [1 if int(value) != 0 else 0 for value in attention_mask]
        if not input_ids or not any(attention_mask):
            raise ValueError(f"{side} row {row_number} is empty after truncation")

        lowest, highest = min(input_ids), max(input_ids)
        if lowest < 0 or highest >= len(tokenizer):
            raise ValueError(
                f"{side} row {row_number} has token IDs [{lowest}, {highest}], "
                f"outside tokenizer vocabulary [0, {len(tokenizer) - 1}]"
            )
        features.append(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )

    padded = tokenizer.pad(features, padding=True, return_tensors="pt")
    mask = padded["attention_mask"].to(dtype=torch.long)
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{side} attention mask is not binary after padding")
    return padded["input_ids"].to(dtype=torch.long), mask


def collate_fn(tokenizer, batch, max_length):
    """Token-aware pair collation with correct mask padding for any pad token ID."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    chosen_ids, chosen_mask = _pair_features(
        tokenizer, batch, "chosen", max_length
    )
    rejected_ids, rejected_mask = _pair_features(
        tokenizer, batch, "rejected", max_length
    )
    return {
        "input_ids_chosen": chosen_ids,
        "attention_mask_chosen": chosen_mask,
        "input_ids_rejected": rejected_ids,
        "attention_mask_rejected": rejected_mask,
    }


def _set_active_adapter(model, name):
    (model.module if hasattr(model, "module") else model).set_adapter(name)


def _model_dtype(args):
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def _xgboost_device_params(tree_method):
    """Translate legacy gpu_hist to the XGBoost 2.x device API."""
    major_version = int(xgb.__version__.split(".", 1)[0])
    if tree_method == "gpu_hist" and major_version >= 2:
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": tree_method}


def main():
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(
                timeout=timedelta(minutes=args.distributed_timeout_minutes)
            )
        ]
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    checkpoint_root = resolve_checkpoint_root(
        args.checkpoint_dir, args.adapter_glob
    )
    adapters = discover_adapter_checkpoints(checkpoint_root)
    tokenizer = load_checkpoint_tokenizer(
        checkpoint_root, args.base_model, args.max_length
    )
    dtype = _model_dtype(args)

    model_params = {
        "vhead_layer_type": args.layer_type,
        "vhead_num_neurons": args.num_neurons,
        "vhead_num_layers": args.num_layers,
    }
    base = AutoModelForCausalLMWithMultiValueHead.from_pretrained(
        args.base_model,
        device_map=device,
        torch_dtype=dtype,
        num_value_heads=len(adapters),
        attn_implementation=args.attn_implementation,
        **model_params,
    )
    base.pretrained_model.resize_token_embeddings(len(tokenizer))
    base.config.pad_token_id = tokenizer.pad_token_id
    model = load_adapters_and_heads(base, adapters, device, dtype)
    adapter_names = [adapter.name for adapter in adapters]
    if is_main:
        print(
            f"Loaded {len(adapters)} adapters from {checkpoint_root}; "
            f"padding_side={tokenizer.padding_side}, dtype={dtype}"
        )

    # Keep the Arrow/datasets dependency lazy so checkpoint and collator
    # utilities remain importable in lightweight CPU environments.
    from load_datasets import load_train_eval_dataset

    train_ds, _ = load_train_eval_dataset(
        args.dataset,
        tokenizer,
        size=args.max_train_samples,
        mode=args.dataset_mode,
        load_eval=False,
    )
    _, eval_ds = load_train_eval_dataset(
        args.eval_dataset,
        tokenizer,
        size=args.max_eval_samples,
        mode=args.dataset_mode,
        load_train=False,
    )
    if args.max_train_samples is not None:
        train_ds = train_ds.select(
            range(min(args.max_train_samples, len(train_ds)))
        )
    if args.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    if is_main:
        print(f"Dataset sizes: train={len(train_ds)}, validation={len(eval_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_fn(tokenizer, rows, args.max_length),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_fn(tokenizer, rows, args.max_length),
    )
    train_loader, eval_loader = accelerator.prepare(train_loader, eval_loader)

    def gather_pairwise_features(loader, description):
        feature_rows, labels, group_sizes = [], [], []
        model.eval()

        for batch in tqdm(loader, disable=not is_main, desc=description):
            with torch.inference_mode():
                chosen_columns, rejected_columns = [], []
                for head_index, adapter_name in enumerate(adapter_names):
                    _set_active_adapter(model, adapter_name)
                    chosen_reward = model(
                        batch["input_ids_chosen"],
                        attention_mask=batch["attention_mask_chosen"],
                        active_head=head_index,
                    )[-1].reshape(-1)
                    rejected_reward = model(
                        batch["input_ids_rejected"],
                        attention_mask=batch["attention_mask_rejected"],
                        active_head=head_index,
                    )[-1].reshape(-1)
                    chosen_columns.append(chosen_reward)
                    rejected_columns.append(rejected_reward)

                chosen_matrix = torch.stack(chosen_columns, dim=1)
                rejected_matrix = torch.stack(rejected_columns, dim=1)
                # Gather while dimension 0 still means dataset pairs. Accelerate
                # can then remove duplicated tail samples correctly on the last
                # distributed batch. Gathering after flattening to 2*B rows
                # would apply the pair remainder to reward rows and corrupt
                # XGBoost group boundaries.
                gathered_pairs = accelerator.gather_for_metrics(
                    torch.stack([chosen_matrix, rejected_matrix], dim=1)
                )
                pair_features = gathered_pairs.reshape(-1, len(adapters))
                pair_labels = torch.tensor(
                    [1, 0], device=device, dtype=torch.long
                ).repeat(gathered_pairs.shape[0])

                feature_rows.append(pair_features)
                labels.append(pair_labels)
                group_sizes.extend([2] * gathered_pairs.shape[0])

        if not feature_rows:
            raise RuntimeError(f"{description} produced no feature batches")

        features = torch.cat(feature_rows)
        gathered_labels = torch.cat(labels)
        groups = torch.tensor(group_sizes, device=device, dtype=torch.int32)
        return (
            features.float().cpu().numpy(),
            gathered_labels.cpu().numpy(),
            groups.cpu().numpy(),
        )

    train_features, train_labels, train_groups = gather_pairwise_features(
        train_loader, "extract train"
    )
    valid_features, valid_labels, valid_groups = gather_pairwise_features(
        eval_loader, "extract validation"
    )

    if is_main:
        dtrain = xgb.DMatrix(train_features, label=train_labels)
        dtrain.set_group(train_groups)
        dvalid = xgb.DMatrix(valid_features, label=valid_labels)
        dvalid.set_group(valid_groups)

        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@2",
            "disable_default_eval_metric": 1,
            "max_depth": args.xgb_max_depth,
            "min_child_weight": args.xgb_min_child_weight,
            "gamma": args.xgb_gamma,
            "lambda": args.xgb_reg_lambda,
            "alpha": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.8,
            "eta": args.xgb_eta,
            **_xgboost_device_params(args.tree_method),
        }
        print("Training XGBoost")
        pair_callback = PairAccCB(
            dvalid,
            checkpoint_path=args.booster_out,
            early_stopping=args.early_stopping,
        )
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=args.num_round,
            evals=[(dvalid, "validation")],
            callbacks=[pair_callback],
            verbose_eval=True,
        )
        print(
            f"Finished XGBoost: best_iteration={pair_callback.best_iteration}, "
            f"best_pair_accuracy={pair_callback.best_acc:.4f}"
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
