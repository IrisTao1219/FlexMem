#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig
import types
from llava.constants import IGNORE_INDEX
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import repeat_kv, apply_rotary_pos_emb
import math
import warnings

from transformers.utils import logging
logger = logging.get_logger(__name__)


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        cfg=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, input_lengths) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
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
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            if cfg and cfg.generate_method == 'ppl':
                outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                bsz = labels.shape[0]
                logits = outputs.logits[:, -(cfg.n_ctx-cfg.n_prompt):]
                labels = labels[:, -(cfg.n_ctx-cfg.n_prompt):]
                token_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.flatten(),
                    reduction="none",
                )
                gold_score = token_loss.view(bsz, -1)
                z = (labels.view(bsz, -1) > -1).sum(dim=-1)
                gold_score = -gold_score.sum(dim=-1) / z # avg loss value

                return outputs, gold_score.cpu().tolist()[0]
            
            elif cfg and cfg.generate_method == 'attn_score':
                outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                
                scores = self.get_selfattn_scores(
                    cfg.n_ctx,
                    labels=labels,
                    mode=cfg.mode,
                    input_lengths=input_lengths,
                    # mask_query=mask_query,
                )     

                return outputs, scores

            else:
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
    
    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in super().get_decoder().layers:
            mod.self_attn.score_storage = None
            mod.self_attn.normalized_score_storage = None
            mod.self_attn.prob_storage = None

    def overwrite_forward_selfattn(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in super().get_decoder().layers:
            xattn = mod.self_attn
            xattn.forward = types.MethodType(self_attn_forward, xattn)

    def create_selfattn_storage(self):
        for mod in super().get_decoder().layers:
            xattn = mod.self_attn
            xattn.score_storage = None
            xattn.normalized_score_storage = None
            xattn.prob_storage = None

    @torch.no_grad()
    def get_selfattn_scores(self, n_passages, labels, mode="norms", mask_query=None, input_lengths=None):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        
        scores: [bsz, l_visual+l_text, l_visual+l_text]
        """
        scores, norms, probs = [], [], []
        for mod in super().get_decoder().layers:
            scores.append(mod.self_attn.score_storage)
            norms.append(mod.self_attn.normalized_score_storage)
            probs.append(mod.self_attn.prob_storage)
        scores = torch.stack(scores)
        norms = torch.stack(norms)
        probs = torch.stack(probs) # [num_layers, bsz, l_total, l_total]

        output = {}
        if "scores" in mode or "all" in mode:
            self.aggregate_value(scores, labels, n_passages, input_lengths, mask_query, output, prefix="scores")
        if "probs" in mode or "all" in mode:
            self.aggregate_value(probs, labels, n_passages, input_lengths, mask_query, output, prefix="probs")
        if "norms" in mode or "all" in mode:
            self.aggregate_value(norms, labels, n_passages, input_lengths, mask_query, output, prefix="norms")
        return output

    def aggregate_value(self, scores, labels, n_ctx, input_lengths, mask_query=None, output={}, prefix=""):

        scores = scores[:-1].to(torch.float32) # TODO: 什么 Qwen2 模型中最后一层的 self-attn-scores 中会有 nan 值

        n_layers, bsz = scores.shape[:2]
        scores = scores.sum(dim=[0]) # [bsz, n_tokens, n_tokens]

        # scores = scores.view(n_layers, bsz, n_tokens, n_ctx, -1) # [num_layers, bsz, query_len, ctx_num, ctx_len]

        scores_first = torch.ones((bsz, n_ctx), dtype=scores.dtype, device=scores.device)
        scores_sum = torch.ones((bsz, n_ctx), dtype=scores.dtype, device=scores.device)

        for i in range(bsz):
            cur_prefix, cur_visual, cur_qa = input_lengths[i]
            cur_ans = (~(labels[i] == IGNORE_INDEX)).sum()
            cur_qs = cur_qa - cur_ans
            cur_score = scores[i, cur_prefix+cur_visual:sum(input_lengths[i]), cur_prefix:cur_prefix+cur_visual] # [n_qa, n_visual]

            ctx_len = cur_visual // n_ctx
            cur_score = cur_score.contiguous().view(cur_qa, n_ctx, ctx_len) # [qa_len, n_ctx, ctx_len]

            # ntokens_first = n_layers * (cur_qs+1) * ctx_len
            # ntokens_sum = n_layers * (cur_qs+cur_ans) * ctx_len
            ntokens_first = n_layers * 1 * ctx_len
            ntokens_sum = n_layers * cur_ans * ctx_len

            print(f'cur_score: {cur_score[cur_qs:].sum(dim=[0, 2])}, {ntokens_sum}')

            cur_score_first = cur_score[cur_qs:cur_qs+1].sum(dim=[0, 2]) / ntokens_first
            cur_score_sum = cur_score[cur_qs:].sum(dim=[0, 2]) / ntokens_sum

            scores_first[i] = cur_score_first
            scores_sum[i] = cur_score_sum

            print(f"cur_score_sum: {cur_score_sum}")

        output[f"{prefix}first"] = scores_first
        output[f"{prefix}sum"] = scores_sum # evalnormsum
        # output[f"{prefix}avg"] = scores_wquery / ntokens_wquery

        scores_woquery = None
        # Compute scores based on scores without query
        # if not mask_query is None:
        #     output[f"{prefix}woquery"] = self.get_woquery_score(scores, mask_query, mask, labels, n_layers)

        return output

def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if scores.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {scores.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

        scores = scores + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if hasattr(self, "score_storage"):
        with torch.no_grad():
            self.score_storage = scores.detach().mean(dim=1)
            self.prob_storage = attn_weights.detach().mean(dim=1)
            self.normalized_score_storage = (
                (torch.norm(value_states.float(), dim=-1)[:, :, None] * attn_weights).detach().mean(dim=1) # bn1k * bnqk -> bqk
            )


    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
