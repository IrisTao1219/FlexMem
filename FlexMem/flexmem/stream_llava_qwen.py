from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .modeling_qwen2 import Qwen2Model 
from transformers import Qwen2Config, Qwen2ForCausalLM

import types
from llava.constants import IGNORE_INDEX
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import repeat_kv, apply_rotary_pos_emb
import math
import warnings

from .modeling_memory import Memory
from llava.model.utils import optional_grad_ctx, BeaconModelOutput

from transformers.utils import logging
logger = logging.get_logger(__name__)


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class Stream_LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained to extend vocab size according to beacon_size."""
        kwargs.update(output_loading_info=True)
        tokens_per_frame , chunk_size , topb , preblk = kwargs.pop("tokens_per_frame", None),kwargs.pop("chunk_size", None) , kwargs.pop("topb", None) , kwargs.pop("preblk", None)
        preratio, decratio = kwargs.pop("preratio", None) , kwargs.pop("decratio", None)
        
        model, loading_info = super().from_pretrained(*args, **kwargs)
        
        
        # NOTE: set memory after from_pretrained because there may be another transformer model inside the Memory object, which may cause weird erros during loading
        config = model.config
        config.beacon_parallel_window = 1
        config.beacon_window = tokens_per_frame * chunk_size
        config.beacon_stride = tokens_per_frame * chunk_size
        config.beacon_sink_size = 0
        config.beacon_ratio = [0]
        config.beacon_pos = 'visual_iterative' 
        config.beacon_parallel_window = 1
        config.beacon_attn = "full-coverage"
        config.append_question = True
        config.topk_clips = topb
        model.config = config
        model.memory = Memory(
            model_config=config,
            k_seq_dim=2,
            v_seq_dim=2,
        )
        model.memory.tokens_per_frame = tokens_per_frame
        model.memory.preratio = preratio
        model.memory.decratio = decratio
        model.memory.preblk = preblk
        return model
    

    def _native_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[bool] = True,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_features: Optional[torch.Tensor] = None,
        preratio: int = 1,
        decratio: int = 1,
    ) -> Union[Tuple, BeaconModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if past_key_values is None:
            # NOTE: set beacon size to 0 to avoid using any beacon parameters, see Qwen2Attention.forward
            past_key_values = [(None, None, 0, None) for _ in range(self.config.num_hidden_layers)]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_features=image_features,
            preratio = preratio,
            decratio = decratio,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        batch_loss = None
        valid_token_num = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return BeaconModelOutput(
            loss=loss,
            batch_loss=batch_loss,
            valid_token_num=valid_token_num,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _beacon_forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        question_ids: Optional[torch.Tensor] = None,
        image_features:Optional[torch.Tensor] = None,
        is_parallel: Optional[bool] = False,
        qid: Optional[str] = None,
    ):
        tokens_per_frame = self.memory.tokens_per_frame
        self.memory.prepare(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            skip_first=beacon_skip_first,
            skip_last=beacon_skip_last,
            question_ids=question_ids,
            imgs_len=image_features[0].shape[0]*tokens_per_frame
        )

        beacon_window = int(self.config.beacon_window / tokens_per_frame)
        images, modalities, image_sizes = image_features # NOTE: here image_sizes is None
        # after the first window, one token at a time
        step_num = 0
        while not self.memory.finish:
            step_num += 1

            input_ids, attention_mask, position_ids, past_key_values, labels = self.memory.step()

            if past_key_values and past_key_values[0][2] == -1:
                image_features = self.get_image_features([images[:beacon_window]], modalities, image_sizes)
                image_features=torch.stack(image_features).squeeze(0)
                images = images[beacon_window:]

            preratio , decratio = self.memory.preratio, self.memory.decratio
            
            if is_parallel and past_key_values[0][2] == -1:
                cur_inputs_embeds = image_features.reshape(-1, self.config.beacon_window, image_features.shape[-1])
                blk_size = int(image_features.shape[0] // self.config.beacon_window)

                position_ids = position_ids.expand(blk_size, -1)
                cur_past_key_values = []
                for i in range(self.config.num_hidden_layers):
                    past_key, past_value, beacon_size, beacon_indices = past_key_values[i]
                    past_key = past_key.expand(blk_size, -1, -1, -1)
                    past_value = past_value.expand(blk_size, -1, -1, -1)
                    cur_past_key_values.append((past_key, past_value, beacon_size, beacon_indices))
                past_key_values = cur_past_key_values

                outputs = self._native_forward(
                    # input_ids=input_ids,
                    inputs_embeds=cur_inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    past_key_values=past_key_values,
                    labels=labels,
                    # NOTE: the labels have been shifted so that all tokens in the window have the proper loss
                    shift_labels=False,
                    image_features=image_features,
                    preratio = preratio,
                    decratio = decratio,
                )

            else:
                outputs = self._native_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    labels=labels,
                    # NOTE: the labels have been shifted so that all tokens in the window have the proper loss
                    shift_labels=False,
                    image_features=image_features,
                    preratio = preratio,
                    decratio = decratio,
                )


            # update past_key_values
            self.memory.update_memory(qid, outputs.past_key_values, step_num)


            if labels is not None:
                # update loss
                self.memory.update_loss(outputs.batch_loss, outputs.valid_token_num)

        outputs = self.memory.output(outputs)

        # input()

        return outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],

        image_features: Optional[torch.FloatTensor] = None,
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        question_ids: Optional[torch.Tensor] = None,
        is_parallel: Optional[bool] = False,
        qid: Optional[str] = None,

        dpo_forward: Optional[bool] = False,
        cache_position=None,
        cfg=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        num_tokens=image_features[0].shape[0]*self.memory.tokens_per_frame

        if -200 in input_ids:
            start_value = -200
            if num_tokens !=0:
                insert_index = (input_ids == start_value).nonzero(as_tuple=True)[1][0].item()
                negative_tokens = torch.arange(start_value, start_value - num_tokens, -1, device=input_ids.device)
                if labels !=None:
                    ignore_labels = torch.full((1, num_tokens), -100, device=labels.device, dtype=labels.dtype)
                    before_labels = labels[:, :insert_index]
                    after_labels = labels[:, insert_index + 1:]
                    labels = torch.cat((before_labels, ignore_labels, after_labels), dim=1)

                before_input_ids = input_ids[:, :insert_index]
                after_input_ids = input_ids[:, insert_index + 1:]
                input_ids = torch.cat((before_input_ids, negative_tokens.unsqueeze(0), after_input_ids), dim=1)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            input_ids[input_ids < 0] = 151646

        if beacon_skip_first is None:
            beacon_skip_first=14
            beacon_skip_last=beacon_skip_first + num_tokens

        with optional_grad_ctx(with_grad=self.training):
            return self._beacon_forward(input_ids,
                                        attention_mask,
                                        position_ids,
                                        past_key_values,
                                        inputs_embeds,
                                        labels,
                                        use_cache,
                                        output_attentions,
                                        output_hidden_states,
                                        return_dict,
                                        beacon_skip_first,
                                        beacon_skip_last,
                                        question_ids,
                                        image_features,
                                        is_parallel,
                                        qid)
            


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        question_ids: Optional[torch.Tensor] = None,
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        is_parallel: Optional[bool] = False,
        qid: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        image_features = [images[0], modalities, image_sizes]
        num_tokens = images[0].shape[0]*self.memory.tokens_per_frame

        if beacon_skip_first is None or beacon_skip_last is None:
            beacon_skip_first = (inputs == -200).nonzero(as_tuple=True)[1].item()
            beacon_skip_last = beacon_skip_first  + num_tokens

        if -200 in inputs: # IMAGE_TOKEN_INDEX == -200
            start_value = -200
            input_ids=inputs
            if num_tokens !=0:
                insert_index = (input_ids == start_value).nonzero(as_tuple=True)[1][0].item()
                negative_tokens = torch.arange(start_value, start_value - num_tokens, -1, device=input_ids.device)
                before_input_ids = input_ids[:, :insert_index]
                after_input_ids = input_ids[:, insert_index + 1:]
                input_ids = torch.cat((before_input_ids, negative_tokens.unsqueeze(0), after_input_ids), dim=1)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                input_ids[input_ids < 0] = self.config.vocab_size-1 
                inputs=input_ids
       
        return super().generate(position_ids=position_ids, attention_mask=attention_mask,inputs=inputs,beacon_skip_first=beacon_skip_first, beacon_skip_last= beacon_skip_last, question_ids=question_ids, image_features=image_features, is_parallel=is_parallel,qid=qid, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, question_ids=None, beacon_skip_first=None, beacon_skip_last=None, is_parallel=False,qid=None, **kwargs):

        if past_key_values:
            input_ids = input_ids[:, -1:] # TODO: used for decoding

        
        model_inputs = {
            "input_ids": input_ids,
            "question_ids": question_ids,
            "beacon_skip_first": beacon_skip_first,
            "beacon_skip_last": beacon_skip_last,
            "is_parallel": is_parallel,
            "qid": qid,
        }

        if 'image_features' in kwargs:
            model_inputs["image_features"] = kwargs['image_features']
        
        return model_inputs
    
    




AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, Stream_LlavaQwenForCausalLM)
