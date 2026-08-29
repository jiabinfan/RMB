from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import numpy as np
import torch
import torch.nn as nn
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer



@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        margins = []
        has_preference_targets = any(
            "preference_target" in feature for feature in features
        )
        preference_targets = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if 'margin' in feature.keys():
                margins.append(feature['margin'])
            if has_preference_targets:
                # Mixed RRM batches may contain original pairs (target=1) and
                # neutral cross-context pairs (target=0.5).
                preference_targets.append(
                    float(feature.get("preference_target", 1.0))
                )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "margin": margins,
        }
        # Preserve the exact legacy batch schema for every pre-RRM dataset.
        if has_preference_targets:
            batch["preference_target"] = torch.tensor(
                preference_targets, dtype=torch.float32
            )
        return batch


class SimpleRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):
        self.loss_type = kwargs.pop('loss_type', 'bt')
        self.weight_ratio = kwargs.pop('weight_ratio', 0.1)
        super(SimpleRewardTrainer, self).__init__(**kwargs)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        if self.loss_type == 'bt':
            target = inputs.get("preference_target")
            if target is None:
                loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            else:
                target = target.to(
                    device=rewards_j.device, dtype=rewards_j.dtype
                ).reshape_as(rewards_j)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    rewards_j - rewards_k, target
                )
        elif self.loss_type == 'pos_reg':
            loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean() - self.weight_ratio * nn.functional.logsigmoid(rewards_j.mean())
        elif self.loss_type == 'margin':
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k - torch.tensor(inputs["margin"], device=inputs["margin"][0].device).view(-1,1)).mean()
        elif self.loss_type == 'labelsmooth':
            loss = - (1-self.weight_ratio) * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - self.weight_ratio * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
        else:
            raise NotImplementedError

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
