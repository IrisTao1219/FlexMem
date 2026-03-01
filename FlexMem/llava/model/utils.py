from transformers import AutoConfig
import torch

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Mapping, Optional, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPast


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def patch_llama(method='stream'):
    pass

def optional_grad_ctx(with_grad=False):
    if with_grad:
        return nullcontext()
    else:
        return torch.no_grad()
    
@dataclass
class BeaconModelOutput(BaseModelOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    batch_loss: Optional[torch.FloatTensor] = None
    valid_token_num: Optional[torch.LongTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None