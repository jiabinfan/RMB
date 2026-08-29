import sys
import tempfile
import unittest
from pathlib import Path

import torch

REWARD_MODELS = Path(__file__).resolve().parents[1] / "reward_models"
sys.path.insert(0, str(REWARD_MODELS))

from grm_utils import last_non_pad_indices
from hsic_rms_train import hsic_diversity_penalty
from run_booster_rmb import collate_fn, discover_adapter_checkpoints


class RmbCoreTests(unittest.TestCase):
    class FakeTokenizer:
        pad_token_id = 7
        truncation_side = "right"

        def __len__(self):
            return 16

        def pad(self, features, padding=True, return_tensors="pt"):
            width = max(len(feature["input_ids"]) for feature in features)
            ids = []
            masks = []
            for feature in features:
                amount = width - len(feature["input_ids"])
                ids.append(feature["input_ids"] + [self.pad_token_id] * amount)
                masks.append(feature["attention_mask"] + [0] * amount)
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(masks),
            }

    def test_last_non_pad_indices_supports_both_padding_directions(self):
        mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1],
                [1, 0, 1, 0, 0],
            ]
        )

        actual = last_non_pad_indices(mask, 3, 5, mask.device)

        self.assertEqual(actual.tolist(), [2, 4, 2])

    def test_normalized_hsic_detects_duplicate_margins_and_has_gradients(self):
        generator = torch.Generator().manual_seed(7)
        base = torch.randn(96, 1, generator=generator)
        duplicate = torch.cat([base, base], dim=1).requires_grad_()
        independent = torch.randn(96, 2, generator=generator)

        duplicate_penalty = hsic_diversity_penalty(duplicate)
        independent_penalty = hsic_diversity_penalty(independent)
        duplicate_penalty.backward()

        self.assertGreater(duplicate_penalty.item(), 0.99)
        self.assertGreater(duplicate_penalty.item(), independent_penalty.item())
        self.assertIsNotNone(duplicate.grad)
        self.assertTrue(torch.isfinite(duplicate.grad).all())

    def test_checkpoint_discovery_accepts_nested_and_flat_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, nested in enumerate((True, False)):
                outer = root / f"adapter_{index}"
                payload = outer / outer.name if nested else outer
                payload.mkdir(parents=True)
                (payload / "adapter_config.json").write_text(
                    "{}", encoding="utf-8"
                )
                (payload / "adapter_model.safetensors").touch()
                (outer / "v_head.bin").touch()

            adapters = discover_adapter_checkpoints(root)

            self.assertEqual([adapter.index for adapter in adapters], [0, 1])
            self.assertEqual(adapters[0].weights_dir.name, "adapter_0")
            self.assertEqual(adapters[0].weights_dir.parent.name, "adapter_0")
            self.assertEqual(adapters[1].weights_dir, root / "adapter_1")

    def test_pair_collator_never_uses_pad_token_as_attention(self):
        rows = [
            {
                "input_ids_chosen": [2, 3],
                "attention_mask_chosen": [1, 1],
                "input_ids_rejected": [4],
                "attention_mask_rejected": [1],
            },
            {
                "input_ids_chosen": [5],
                "attention_mask_chosen": [1],
                "input_ids_rejected": [6, 2],
                "attention_mask_rejected": [1, 1],
            },
        ]

        batch = collate_fn(self.FakeTokenizer(), rows, max_length=4)

        self.assertEqual(
            batch["attention_mask_chosen"].tolist(), [[1, 1], [1, 0]]
        )
        self.assertEqual(
            batch["attention_mask_rejected"].tolist(), [[1, 0], [1, 1]]
        )


if __name__ == "__main__":
    unittest.main()
