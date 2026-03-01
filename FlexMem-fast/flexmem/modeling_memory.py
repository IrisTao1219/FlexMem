
import os
import torch
import time
import numpy as np
import torch.distributed as dist
from transformers.utils import logging
from transformers import AutoTokenizer
from itertools import cycle
from typing import List
import json
import math
from torch import nn

from .modeling_qwen2 import repeat_kv

from scipy.interpolate import interp1d
import joblib
# import lightgbm as lgb
import torch.nn.functional as F

logger = logging.get_logger(__name__)

def minmax_normalize_along_dim(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    a_min = np.min(arr, axis=axis, keepdims=True)
    a_max = np.max(arr, axis=axis, keepdims=True)
    denom = a_max - a_min
    denom = np.where(denom == 0.0, 1.0, denom)
    normed = (arr - a_min) / denom
    normed = np.where(a_max == a_min, 0.0, normed)
    return normed

class Memory(torch.nn.Module):
    def __init__(
        self, 
        model_config, 
        k_seq_dim:int=2, 
        v_seq_dim:int=2, 
    ):
        """Setup necessary attributes."""
        super().__init__()

        self.config = model_config

        # initialize necessary parameters
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.rng = np.random.default_rng(42)

        self._post_validation()
        self.reset()
    
    @property
    def beacon_token(self):
        return self.config.vocab_size

    def _post_validation(self, verbose=True):
        assert self.config.beacon_window >= self.config.beacon_stride, f"Make sure the beacon_window {self.config.beacon_window} >= beacon_stride {self.config.beacon_stride}!"
        if self.config.beacon_pos == "interleave":
            assert self.config.beacon_window == self.config.beacon_stride, f"Make sure the beacon_window equals to beacon_stride when using interleaving mode."
        if self.config.beacon_parallel_window > 1:
            assert self.config._attn_implementation != "flash_attention_2", f"Currently parallel window does not support flash_attention_2!"

        self._cpu = torch.device("cpu")

    def set(self, verbose=True, **kwargs):
        """
        Set attributes out of the constructor.
        """
        for k, v in kwargs.items():
            setattr(self.config, k, v)
        self._post_validation(verbose=verbose)

    def reset(self):
        """Initialize attributes for a new sequence."""
        # the cursor pointing to the start of the current window
        self.start_idx = 0
        # the cursor pointing to the end of the current window
        self.end_idx = 0
        # the beacon sizes of all strides
        self.all_beacon_sizes = []
        # the loss per batch
        self.batch_loss = None
        # the valid token number per batch
        self.valid_token_num = None
        # the step index for processing the input_ids
        self.step_idx = 0
        # used in set_compression_ratio
        self.compression_ratio = None
        # the previous inputs is a full window or not, defaults to True
        self.is_full_window = True
        # the number of raw activations to preserve in update_memory (only useful when beacon_stride < beacon_window)
        self.raw_size_to_cache = 0
        # the length of question ids
        self.qs_len = 0
        self.question_ids = None
        # perform retrieval
        self.retrieval = True

        self.is_decoding = False

        # the number of tokens in previous stride that should be compressed by the upcoming beacon
        self.interleave_remainder = 0
        # compression ratio for the unfinished window
        self.interleave_compression_ratio = None
        self.beacon_indices = None

        self.all_input_ids = None
        self.all_attention_mask = None
        self.all_labels = None

        # NOTE: will be reset in prepare()
        self.beacon_skip_first = None
        self.beacon_skip_last = None

        # the raw activations of recent tokens
        self.raw_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        # the attention sink activations
        self.sink_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        # the beacon activations
        self.beacon_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        # the all past visual activations 
        self.visual_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]

        # the all past visual activations 
        self.compression_activations = [([],[]) for _ in range(self.config.num_hidden_layers)]
        # store visualization information
        self.visualization = [[] for _ in range(self.config.num_hidden_layers)]
        # self.all_selction_scores = None
        # store long term memory
        self.long_term_memory = [(None, None) for _ in range(self.config.num_hidden_layers)]
        
        self.all_selection_scores = None
        self.retrieval_activations = [None for _ in range(self.config.num_hidden_layers)]
        
    def partial_reset(self):
        self.retrieval = True
        self.raw_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        self.beacon_activations = self.long_term_memory
    @property
    def all_sequence_length(self):
        if self.all_input_ids is None:
            return 0
        else:
            return self.all_input_ids.shape[1]

    @property
    def batch_size(self):
        if self.all_input_ids is None:
            return 0
        else:
            return self.all_input_ids.shape[0]

    @property
    def finish(self):
        is_finish = self.end_idx == self.all_sequence_length
        return is_finish

    @property
    def dtype(self):
        return self.config.torch_dtype

    @property
    def min_value(self):
        return torch.finfo(self.dtype).min

    @property
    def max_position_embeddings(self):
        max_position_embeddings = self.config.max_position_embeddings
        if getattr(self.config, "rope_scaling", None) is not None:
            scaling_factor = self.config.rope_scaling["factor"]
            max_position_embeddings = max_position_embeddings * scaling_factor
        return max_position_embeddings

    @property
    def beacon_window(self):
        if (
            self.beacon_skip_last is not None
            and self.start_idx < self.beacon_skip_last
            and self.start_idx + self.config.beacon_window > self.beacon_skip_last
        ):
            return self.beacon_skip_last - self.start_idx
        elif (
            self.beacon_skip_last is not None
            and self.start_idx < self.beacon_skip_last
            and self.start_idx + self.config.beacon_window + 1 == self.beacon_skip_last
        ):
            return self.config.beacon_window + 1
        else:
            return self.config.beacon_window

    @property
    def beacon_stride(self):
        if (
            self.beacon_skip_last is not None
            and self.start_idx < self.beacon_skip_last
            and self.start_idx + self.config.beacon_window > self.beacon_skip_last
        ):
            return self.beacon_skip_last - self.start_idx
        elif (
            self.beacon_skip_last is not None
            and self.start_idx < self.beacon_skip_last
            and self.start_idx + self.config.beacon_window + 1 == self.beacon_skip_last
        ):
            return self.config.beacon_stride + 1
        else:
            return self.config.beacon_stride
            
    def get_memory(self):
        past_key_values = []
        for layer_idx in range(self.config.num_hidden_layers):
            sink_key, sink_value = self.sink_activations[layer_idx]
            beacon_key, beacon_value = self.beacon_activations[layer_idx]
            raw_key, raw_value = self.raw_activations[layer_idx]

            key = cat_tensor([
                sink_key, beacon_key, raw_key,
            ], dim=self.k_seq_dim)
            value = cat_tensor([
                sink_value, beacon_value, raw_value,
            ], dim=self.v_seq_dim)

            layer_past_key_values = (key, value)
            past_key_values.append(layer_past_key_values)
        return past_key_values
 
    def get_memory_size(self):
        """
        Sink memory size, beacon memory size and raw memory size.
        """
        sink_memory_size = 0
        beacon_memory_size = 0
        raw_memory_size = 0
        if self.sink_activations[0][0] is not None:
            sink_memory_size += self.sink_activations[0][0].shape[self.k_seq_dim]
        if self.beacon_activations[0][0] is not None:
            beacon_memory_size += self.beacon_activations[0][0].shape[self.k_seq_dim]
        if self.raw_activations[0][0] is not None:
            raw_memory_size += self.raw_activations[0][0].shape[self.k_seq_dim]
        return sink_memory_size, beacon_memory_size, raw_memory_size

    def prepare(self, input_ids, attention_mask, labels, skip_first=None, skip_last=None, question_ids=None, imgs_len=0):
        """
        Prepare inputs for the model. These inputs belong to the same sequence.
        """
        self._device = input_ids.device

        # accumulate input_ids
        if self.all_input_ids is None:
            self.all_input_ids = input_ids.cpu() # for prefill
        else:
            self.all_input_ids = torch.cat([self.all_input_ids, input_ids.cpu()], dim=1) # for decoding

        # accumulate attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=torch.device("cpu"))
        if self.all_attention_mask is None:
            self.all_attention_mask = attention_mask.cpu()
        else:
            self.all_attention_mask = torch.cat([self.all_attention_mask, attention_mask.cpu()], dim=1)

        # accumulate labels if exisits
        if labels is not None:
            # rotate labels in advance so that the loss of the last token is not ignored in every window
            labels = torch.cat([labels[:, 1:].cpu(), torch.tensor([-100]).expand(labels.shape[0], 1)], dim=1)
            if self.all_labels is None:
                self.all_labels = labels.cpu()
            else:
                self.all_labels = torch.cat([self.all_labels, labels], dim=1)
            assert self.all_input_ids.shape[1] == self.all_labels.shape[1], f"Found inconsistent all_input_ids {self.all_input_ids.shape} and all_labels {self.all_labels.shape}!"
        
        if skip_first is not None:
            assert self.config.beacon_parallel_window == 1, f"Make sure the parallel window is set to 1 when using beacon_skip!"
            assert self.config.beacon_window == self.config.beacon_stride, f"Make sure the beacon_window equals to beacon_stride when using beacon_skip."
            assert self.config.beacon_sink_size == 0, f"Make sure the beacon_sink_size is set to 0 when using beacon_skip!"
        # stop compression after how many tokens
        if skip_last is not None:
            skip_first = skip_first if skip_first is not None else 0
            assert self.config.beacon_sink_size == 0, "Make sure the beacon_sink_size is zero when using skip_last!"

        self.question_ids = question_ids
        qs_len = question_ids.shape[1] if question_ids is not None else 0

        self.beacon_skip_first = skip_first
        self.beacon_skip_last = skip_last

        self.qs_len = qs_len
        
    def set_compression_ratio(self, start_idx, end_idx):
        """Choose a condensing ratio from self.config.beacon_ratio"""
        def filter_ratio(ratios, stride):
            valid_ratios = []
            for ratio in ratios:
                # stride must be bigger than condensing ratio because we there must be at least one beacon
                if stride < ratio:
                    continue
                # the stride must be evenly divisible by condensing ratio
                if ratio > 0 and (stride % ratio) != 0:
                    continue
                # when training, ratio=0 is valid if previous windows contain beacon or later windows contain beacon
                if ratio == 0 and self.training:
                    previous_has_zero = -1 in self.all_beacon_sizes
                    following_has_nonzero = (start_idx + stride + self.beacon_window) <= self.all_sequence_length
                    if previous_has_zero or (not following_has_nonzero):
                        continue
                valid_ratios.append(ratio)
            assert len(valid_ratios), f"Cannot find valid condensing ratio (among {ratios}) for stride {stride}!"
            return valid_ratios

        def get_max_length(ratios):
            max_lengths = []
            for compression_ratio in ratios:
                if compression_ratio > 0:
                    # NOTE: here we must use the scaled position embeddings
                    max_lengths.append((self.max_position_embeddings - self.beacon_window) * compression_ratio + self.beacon_window)
                else:
                    max_lengths.append(self.max_position_embeddings)
            return max_lengths

        if len(self.config.beacon_ratio) == 1:
            return self.config.beacon_ratio[0]

        ratio_mix = self.config.beacon_ratio_mix

        beacon_ratio = filter_ratio(self.config.beacon_ratio, self.beacon_stride)

        if ratio_mix == "instance-random":
            if self.compression_ratio is None:
                beacon_ratio = self.rng.choice(beacon_ratio).tolist()
                self.compression_ratio = beacon_ratio
            else:
                beacon_ratio = self.compression_ratio

        elif ratio_mix == "step-random":
            beacon_ratio = self.rng.choice(beacon_ratio).tolist()
        
        elif ratio_mix == "sequence":
            if self.compression_ratio is None:
                self.compression_ratio = cycle(beacon_ratio)
            beacon_ratio = next(self.compression_ratio)

        elif "adapt" in ratio_mix:
            if self.compression_ratio is None:
                future_length = int(ratio_mix.split("-")[1])
                sequence_length = self.all_input_ids.shape[1] + future_length
                max_lengths = get_max_length(beacon_ratio)
                # ascendingly sort the max lengths
                valid_max_lengths_and_indices = [x for x in enumerate(max_lengths) if x[1] >= sequence_length]
                if len(valid_max_lengths_and_indices):
                    minimum_length_index = min(valid_max_lengths_and_indices, key=lambda x: x[1])[0]
                    # use the minimal possible length for this sequence (the smallest fold ratio)
                    beacon_ratio = beacon_ratio[minimum_length_index]
                else:
                    beacon_ratio = max(beacon_ratio)
                self.compression_ratio = beacon_ratio
            else:
                beacon_ratio = self.compression_ratio

        return beacon_ratio

    def step(self):
        if (
            self.config.beacon_parallel_window > 1 
            and self.config.beacon_stride == self.config.beacon_window
            and 0 not in self.config.beacon_ratio
            and self.all_input_ids[:, self.end_idx:].shape[1] >= self.config.beacon_parallel_window * self.config.beacon_window
        ):
            input_ids_list = []
            attention_mask_list = []
            position_ids_list = []
            labels_list = []

            beacon_size_list = []
            beacon_indices_list = []

            for i in range(self.config.beacon_parallel_window):
                if i == 0:
                    _input_ids, _attention_mask, _position_ids, _past_key_values, _labels = self._step()
                else:
                    _input_ids, _attention_mask, _position_ids, _past_key_values, _labels = self._step(ignore_memory=True)

                input_ids_list.append(_input_ids)
                attention_mask_list.append(_attention_mask)
                position_ids_list.append(_position_ids)
                labels_list.append(_labels)
                beacon_size_list.append(_past_key_values[0][2])
                beacon_indices_list.append(_past_key_values[0][3])

                if i == 0:
                    past_key_values = _past_key_values
                    if past_key_values[0][0] is None:
                        mem_size = 0
                    else:
                        mem_size = past_key_values[0][0].shape[self.k_seq_dim]

                else:
                    # no memory
                    assert _past_key_values[0][0] is None
            
            batch_size = self.all_input_ids.shape[0]
            # NOTE: we do not need to repliace beacon tokens for the last window
            seq_len = sum(x.shape[1] for x in input_ids_list) + sum(beacon_size_list) - beacon_size_list[-1]

            input_ids = _input_ids.new_zeros((batch_size, seq_len)) + self.beacon_token
            # all 0
            attention_mask = _attention_mask.new_zeros((batch_size, 1, seq_len, mem_size + seq_len)) + self.min_value
            position_ids = torch.arange(mem_size + seq_len, device=self._device).expand(batch_size, mem_size + seq_len)
            # 2 indicates the beacon token is used for replication
            beacon_indices = beacon_indices_list[0].new_zeros(seq_len) + 2
            if _labels is not None:
                # -100 because no loss on beacon tokens
                labels = _labels.new_zeros((batch_size, seq_len)) - 100
            else:
                labels = None

            start_idx = 0
            position_offset = mem_size
            for i in range(self.config.beacon_parallel_window):
                beacon_size = beacon_size_list[i]
                _input_ids = input_ids_list[i]
                cur_seq_len = _input_ids.shape[1]
                input_ids[:, start_idx: start_idx + cur_seq_len] = _input_ids
                _attention_mask = attention_mask_list[i]
                _position_ids = position_ids_list[i]
                if i == 0:
                    _attention_mask = _attention_mask[:, :, :, mem_size:]
                    _position_ids = _position_ids[:, mem_size:] - mem_size

                attention_mask[:, :, start_idx: start_idx + cur_seq_len, mem_size + start_idx: mem_size + start_idx + cur_seq_len] = _attention_mask
                position_ids[:, mem_size + start_idx: mem_size + start_idx + cur_seq_len] = _position_ids + position_offset

                # populate beacon_indices
                _beacon_indices = beacon_indices_list[i]
                beacon_indices[start_idx: start_idx + cur_seq_len] = _beacon_indices

                # populate labels
                if labels is not None:
                    # populate labels
                    _labels = labels_list[i]
                    labels[:, start_idx: start_idx + cur_seq_len] = _labels

                # NOTE: when there is sink activations, we need to bias the position_ids for the first window
                if i == 0 and self.config.beacon_sink_size > 0 and self.sink_activations[0][0] is None:
                    position_offset += 1

                # modify the attention and position for replicated beacon tokens
                if i != self.config.beacon_parallel_window - 1:
                    replicate_beacon_row_start = start_idx + cur_seq_len
                    replicate_beacon_col_start = mem_size + start_idx + cur_seq_len
                    # NOTE: any attention mask is okay for replicated beacon tokens, but for convenience we use the causal mask
                    attention_mask[:, :, replicate_beacon_row_start: replicate_beacon_row_start + beacon_size, replicate_beacon_col_start: replicate_beacon_col_start + beacon_size] = _attention_mask.new_full((beacon_size, beacon_size), self.min_value).triu(1)
                    # NOTE: all future tokens can attend to the replicated beacon tokens
                    attention_mask[:, :, replicate_beacon_row_start + beacon_size:, replicate_beacon_col_start: replicate_beacon_col_start + beacon_size] = 0
                    # NOTE: the position of replicated beacon tokens start from 0
                    position_ids[:, mem_size + start_idx + cur_seq_len: mem_size + start_idx + cur_seq_len + beacon_size] = torch.arange(position_offset, position_offset + beacon_size, device=_input_ids.device)[None:]

                start_idx += cur_seq_len + beacon_size
                position_offset += beacon_size

            # the memory is visible to all subsequent tokens
            attention_mask[:, :, :, :max(mem_size, self.config.beacon_sink_size)] = 0

            # NOTE: modify beacon_indices
            for i, (key, value, _, _) in enumerate(past_key_values):
                past_key_values[i] = (key, value, sum(beacon_size_list), beacon_indices)

            # NOTE: update _beacon_indices so that the next-token logits can be properly sliced out in self.output()
            self.beacon_indices = beacon_indices
            
            return input_ids, attention_mask, position_ids, past_key_values, labels

        else:
            return self._step()

    def _step(self, ignore_memory=False):
        """
        Yield inputs for the current sliding window, including the input_ids, attention_mask, position_ids, and past_key_values.
        """
        #============================================#
        # Check whether the inputs fulfills a window.
        #============================================#

        # the starting position of the current window w.r.t. the start of the current input sequence
        start_idx = self.start_idx
        # the end position of the current window w.r.t. the start of the current input sequence
        end_idx = start_idx + self.beacon_window
        # indicates if the current window is completely filled by raw activations and new tokens
        # we only append beacon tokens for full windows
        if end_idx > self.all_sequence_length:
            # the input is shorter than the initial window size
            end_idx = self.all_sequence_length
            is_full_window = False
        else:
            is_full_window = True

        # NOTE: in training, the entire sequence is input to the model at once
        # In the last window, we do not need to append beacons because they will not be used at all
        if self.training and end_idx == self.all_sequence_length:
            next_start_idx = start_idx
            is_full_window = False
            raw_size_to_cache = -1
            beacon_size = 0
            compression_ratio = -1
        
        # NOTE: we do not compress the beacon_skip_first tokens at the beginning of the sequence
        elif self.step_idx == 0 and self.beacon_skip_first is not None:
            end_idx = start_idx + self.beacon_skip_first
            assert end_idx <= self.all_sequence_length
            next_start_idx = end_idx
            is_full_window = True
            raw_size_to_cache = -1
            beacon_size = 0
            compression_ratio = -1
        
        # NOTE: we do not compress tokens after beacon_skip_last tokens
        elif self.beacon_skip_last is not None and start_idx >= self.beacon_skip_last:
            if self.retrieval:
                end_idx = start_idx
                next_start_idx = end_idx
                is_full_window = False
                raw_size_to_cache = -1
                beacon_size = -99 # used for denoting retrieval phrase
                compression_ratio = -1
                self.retrieval = False
            else:
                end_idx = min(start_idx + self.beacon_window, self.all_sequence_length)
                next_start_idx = end_idx
                is_full_window = False
                raw_size_to_cache = -1
                beacon_size = -100 # used for denoting deocoding phrase
                compression_ratio = -1

        else:
            #============================================#
            # Set compression ratio
            #============================================#
        
            if is_full_window:
                beacon_stride = self.beacon_stride
                beacon_size = -1
                next_start_idx = end_idx
                raw_size_to_cache = end_idx - next_start_idx
            else:
                next_start_idx = start_idx
                raw_size_to_cache = -1
                beacon_size = 0
                compression_ratio = 0

        #============================================#
        # Slice out input_ids (raw tokens in the current window)
        #============================================#
        input_ids = self.all_input_ids[:, self.end_idx: end_idx].to(self._device)
        attention_mask = self.all_attention_mask[:, self.end_idx: end_idx].to(self._device)

        if self.config.append_question and beacon_size == -1:
            input_ids = torch.cat([input_ids, self.question_ids.to(self._device)], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(self.question_ids.shape)], dim=1)
        elif beacon_size == -99:
            input_ids = self.question_ids.to(self._device)
            attention_mask = attention_mask.new_ones(self.question_ids.shape)

        if self.all_labels is not None:
            labels = self.all_labels[:, self.end_idx: end_idx].to(self._device)
        else:
            labels = None
        batch_size = input_ids.shape[0]

        #============================================#
        # Insert beacon tokens if necessary.
        #============================================#
        beacon_indices = self.qs_len
        past_key_values = []
        for layer_idx in range(self.config.num_hidden_layers):
            if ignore_memory:
                key, value = None, None
            else:
                sink_key, sink_value = self.sink_activations[layer_idx]
                
                
                if beacon_size == -99:
                    beacon_key, beacon_value = None, None
                else:
                    beacon_key, beacon_value = self.beacon_activations[layer_idx]
                raw_key, raw_value = self.raw_activations[layer_idx]

                key = cat_tensor([
                    sink_key, beacon_key, raw_key,
                ], dim=self.k_seq_dim)
                value = cat_tensor([
                    sink_value, beacon_value, raw_value,
                ], dim=self.v_seq_dim)

            layer_past_key_values = (key, value, beacon_size, beacon_indices)
            past_key_values.append(layer_past_key_values)
        
        #============================================#
        # Prepare attention_mask and position_ids.
        #============================================#
        first_key = past_key_values[0][0]
        mem_size = first_key.shape[self.k_seq_dim] if first_key is not None else 0
        if mem_size > 0:
            attention_mask = torch.cat([attention_mask.new_ones(batch_size, mem_size), attention_mask], dim=1)

        input_length = input_ids.shape[1]
        position_ids = torch.arange(attention_mask.shape[-1], dtype=torch.long, device=self._device).repeat(batch_size, 1) # TODO: here position_ids are always start from scratch.

        if self.config._attn_implementation == "flash_attention_2":
            assert self.config.beacon_attn == "full-coverage", f"Make sure to set beacon_attn='full-coverage' when using flash attention! Found {self.config.beacon_attn}."
            if 0 in attention_mask:
                pass
            else:
                attention_mask = None
        elif self.config._attn_implementation == "sdpa" and self.config.beacon_pos == "append" and beacon_size <= 0 and (input_length == 1 or mem_size == 0):
            attention_mask = None
        else:
            pass

        #============================================#
        # Update necessary attributes.
        #============================================#
        # keep track of whether the current inputs is a full_window
        self.is_full_window = is_full_window
        # keep track of the raw_size_to_cache
        self.raw_size_to_cache = raw_size_to_cache
        # involked in self.output()
        self.all_beacon_sizes.append(beacon_size)
        # update start_idx and end_idx
        # NOTE: the update of start_idx will influence self.beacon_window and self.beacon_stride in case self.beacon_skip_last is not None
        # Therefore, we must make sure all calls to self.beacon_window and self.beacon_stride happen before the update of start_idx
        self.start_idx = next_start_idx
        self.end_idx = end_idx
        self.step_idx += 1

        return input_ids, attention_mask, position_ids, past_key_values, labels

    def update_memory(self, qid, past_key_values, step_num):
        """
        Accumulate beacon activations and raw activations.
        """
        selection_scores = None
        selected_layer_idxs = [14,22,26]
        
        for layer_idx, (key, value, beacon_size, beacon_indices, rotary_emb, long_term_info, retrieval_keys) in enumerate(past_key_values):
            previous_raw_key, previous_raw_value = self.raw_activations[layer_idx]
            previous_compress_key, previous_compress_value = self.compression_activations[layer_idx]
            previous_visual_key, previous_visual_value = self.visual_activations[layer_idx]
            previous_info = self.visualization[layer_idx]


            if self.beacon_skip_first is not None and self.sink_activations[layer_idx][0] is None:
                assert key.shape[self.k_seq_dim] == self.beacon_skip_first
                assert value.shape[self.k_seq_dim] == self.beacon_skip_first
                self.sink_activations[layer_idx] = [
                    key,
                    value,
                ]
                # NOTE: no need to update raw activations and beacon activations as all activations are kept as sink activations
                continue

            if self.beacon_activations[layer_idx][0] is None and self.config.beacon_sink_size > 0:
                # save the sink activations
                # NOTE: we do not slice the key/value activations, which may cause duplication when beacon_ratio=-1 for the first window, but it's okay
                self.sink_activations[layer_idx] = [
                    slice_tensor(key, end=self.config.beacon_sink_size, dim=self.k_seq_dim),
                    slice_tensor(value, end=self.config.beacon_sink_size, dim=self.v_seq_dim),
                ]
            
            if not self.is_full_window:

                if beacon_size == -99:
                    # retrieve relevant chunks based on question
                    if layer_idx <= 2:
                        continue
                    if layer_idx not in selected_layer_idxs:
                        continue
                    
                
                    def cal_crossattn(
                        query, 
                        key, 
                        rope=None,                
                        use_cosine=True,                                                   
                        sys_key=None,
                    ):
                        B, Hq, Q, D = query.shape
                        _, Hkv, K, _ = key.shape
                        S = 5
                        C = K // S  
                        key = key.view(B, Hkv, C, S, D)
                        query = query[:, :, -1:, :]  # [B, Hq, 1, D]
                        
                        if sys_key is not None:
                            sys_key = sys_key.unsqueeze(2).expand(-1, -1, C, -1, -1)  # [B, Hkv, C, K, D]
                            key = torch.cat([sys_key, key], dim=3)  # [B, Hkv, C, K+S, D]
                            S = S + sys_key.shape[3]
                        
                        if rope is not None:
                            q_pos = torch.arange(Q, dtype=torch.long, device=query.device)[None].expand(B, -1)
                            query, _ = rope.new_forward(query, query, q_pos)
                            k_flat = key.reshape(B, Hkv, C * S, D)
                            k_pos = torch.arange(S, dtype=torch.long, device=key.device).repeat(C)  # [0..S-1] * C
                            k_pos = k_pos[None].expand(B, -1)
                            _, k_flat = rope.new_forward(k_flat, k_flat, k_pos)
                            key = k_flat.view(B, Hkv, C, S, D)
                        groups = Hq // Hkv
                        key = key.view(B, Hkv,C*S,D)
                        key = repeat_kv(key, groups)  # [B, Hq, K, D]
                        key = key.view(B, Hq, C, S, D)
                                              
                        q = query.to(torch.float32)
                        k = key.to(torch.float32)

                        logits = None
                        c_step = max(1, min(64, C))
                        for c0 in range(0, C, c_step):
                            c1 = min(C, c0 + c_step)
                            cur_c = c1-c0
                            cur_k = k[:,:, c0:c1,...]
                            if use_cosine:
                                qn = F.normalize(q, p=2, dim=-1)          # [B, Hq, Q, D]
                                kn = F.normalize(cur_k, p=2, dim=-1)  # [B, Hq, C_step, S, D]
                                chunk_logits = torch.einsum("bhqd,bhcsd->bhqcs", qn, kn)  
                            else:
                                chunk_logits = torch.einsum("bhqd,bhcsd->bhqcs", q, cur_k) / math.sqrt(D)
                                chunk_logits = chunk_logits.softmax(dim=-1, dtype=torch.float32) 
                                
                            if sys_key is not None:
                                chunk_logits = chunk_logits[:, :, :, :, sys_key.shape[3]:] 
                                cur_S = S - sys_key.shape[3]
                            

                            chunk_logits = chunk_logits[0].sum(1) # [Hq, C, S]
                            chunk_logits = chunk_logits.reshape(Hkv, -1, c1-c0, cur_S).mean(1) # [num_key_value_heads, chunk_num, chunk_size]
                            chunk_logits = chunk_logits.mean(0).sum(-1) # [C]
                            
                            
                            if logits is None:
                                logits = chunk_logits
                            else:
                                logits = torch.cat([logits, chunk_logits], dim=0) 
                        return logits      
                    
                    sim = cal_crossattn(
                        long_term_info, 
                        self.retrieval_activations[layer_idx], 
                        rope=rotary_emb, 
                        use_cosine=False,
                        sys_key=self.sink_activations[layer_idx][0], 
                    )                 
                    

                    if selection_scores is None:
                        selection_scores = sim
                    else:
                        selection_scores += sim
                else:
                    assert self.raw_size_to_cache == -1
                    raw_key = cat_tensor([
                        previous_raw_key,
                        key
                    ], dim=self.k_seq_dim)
                    raw_value = cat_tensor([
                        previous_raw_value, 
                        value
                    ], dim=self.v_seq_dim)
                    self.raw_activations[layer_idx] = (raw_key, raw_value)

            else:
                # NOTE: use the correct previous_beacon_key and value!
                previous_beacon_key, previous_beacon_value = self.beacon_activations[layer_idx]
                previous_long_term_key, previous_long_term_value = self.long_term_memory[layer_idx]

                previous_retrieval_keys = self.retrieval_activations[layer_idx]

                beacon_key, beacon_value = key, value
                visual_key, visual_value = key, value
                
                self.chunk_size = key.shape[self.k_seq_dim]
                
                retrieval_keys = cat_tensor([
                    previous_retrieval_keys,
                    retrieval_keys
                ], dim=self.k_seq_dim)
                
                
                long_term_key, long_term_value = long_term_info

                # NOTE: long_term memory for encoding 
                long_term_key = cat_tensor([
                    previous_long_term_key, 
                    long_term_key
                ], dim=self.k_seq_dim)
                long_term_value = cat_tensor([
                    previous_long_term_value, 
                    long_term_value
                ], dim=self.v_seq_dim)
                visual_size = int(self.preblk*long_term_info[0].shape[self.k_seq_dim])
                if long_term_key.shape[self.k_seq_dim] > visual_size:
                    long_term_key = slice_tensor(long_term_key, start=-visual_size, dim=self.k_seq_dim)
                    long_term_value = slice_tensor(long_term_value, start=-visual_size, dim=self.v_seq_dim)

                beacon_key, beacon_value = long_term_key, long_term_value
                # NOTE: store all visual tokens
                visual_key = cat_tensor([
                    previous_visual_key, 
                    visual_key
                ], dim=self.k_seq_dim)
                visual_value = cat_tensor([
                    previous_visual_value, 
                    visual_value
                ], dim=self.v_seq_dim)

                self.beacon_activations[layer_idx] = (beacon_key, beacon_value)
                self.visual_activations[layer_idx] = (visual_key, visual_value)
                self.long_term_memory[layer_idx] = (long_term_key, long_term_value)
                self.retrieval_activations[layer_idx] = retrieval_keys

        if beacon_size == -99:
            
            final_selection_scores = selection_scores
            _, keep_indices = selection_scores.topk(min(self.config.topk_clips, len(final_selection_scores)))
            keep_indices = keep_indices.sort().values 
            for layer_idx in range(self.config.num_hidden_layers):
                visual_key, visual_value = self.visual_activations[layer_idx]

                if self.config.topk_clips < len(final_selection_scores):
                    bsz, head_nums, kv_len, dim = visual_key.shape
                    visual_key = visual_key.reshape(bsz, head_nums, -1, self.chunk_size, dim)
                    visual_value = visual_value.reshape(bsz, head_nums, -1, self.chunk_size, dim)
                    visual_key = visual_key[:,:,keep_indices,...].reshape(bsz, head_nums, -1, dim)
                    visual_value = visual_value[:,:,keep_indices,...].reshape(bsz, head_nums, -1, dim)

                beacon_key, beacon_value = visual_key, visual_value 
                self.beacon_activations[layer_idx] = (beacon_key, beacon_value)


    def update_loss(self, batch_loss, valid_token_num):
        """
        Accumulate loss for later perplexity computation and backward pass.
        """
        if self.batch_loss is None:
            # NOTE: multiply valid_token_num because batch_loss is divided by it in advance
            self.batch_loss = batch_loss * valid_token_num
            self.valid_token_num = valid_token_num
        else:
            # NOTE: avoid in-place operations, otherwise there will be gradient errors in training
            self.batch_loss = self.batch_loss + batch_loss * valid_token_num
            self.valid_token_num = self.valid_token_num + valid_token_num

    def output(self, model_outputs):
        """
        Override loss with accumulated loss. Update the next-token logits.
        """
        # override loss
        if self.batch_loss is not None:
            # here the batch_loss is the summation of all token losses in each element
            loss = self.batch_loss.sum() / self.valid_token_num.sum()

            # NOTE: prevent nan
            batch_loss = self.batch_loss / self.valid_token_num
            if (self.valid_token_num == 0).any():
                batch_loss = batch_loss.masked_fill(self.valid_token_num == 0, 0.)

            # NOTE: we must use dict to override values, otherwise trainer cannot find loss
            model_outputs["loss"] = loss
            model_outputs["batch_loss"] = batch_loss

        # override last_hidden_states (used in generation)
        beacon_size = self.all_beacon_sizes[-1]
        # remove logits corresponding to beacon tokens
        if beacon_size > 0:
            logits = model_outputs["logits"]
            beacon_indices = self.beacon_indices[-logits.shape[1]:]
            model_outputs["logits"] = logits[:, beacon_indices == 0]

        return model_outputs

def slice_tensor(x, start=None, end=None, step=None, index=None, dim=2):
    if x is None:
        return None
    if end == 0:
        return None
    if start == x.shape[dim]:
        return None
    if start is not None and start == end:
        return None
    if dim == 2:
        if index is not None:
            return x[:, :, index]
        elif start is None and end is not None:
            if step is None:
                return x[:, :, :end, ...]
            else:
                return x[:, :, :end:step, ...]
        elif start is not None and end is None:
            if step is None:
                return x[:, :, start:, ...]
            else:
                return x[:, :, start::step, ...]
        elif start is not None and end is not None:
            if step is None:
                return x[:, :, start:end, ...]
            else:
                return x[:, :, start:end:step, ...]
    elif dim == 1:
        if index is not None:
            return x[:, :, index]
        elif start is None and end is not None:
            if step is None:
                return x[:, :end, ...]
            else:
                return x[:, :end:step, ...]
        elif start is not None and end is None:
            if step is None:
                return x[:, start:, ...]
            else:
                return x[:, start::step, ...]
        elif start is not None and end is not None:
            if step is None:
                return x[:, start:end, ...]
            else:
                return x[:, start:end:step, ...]
    else:
        raise NotImplementedError

def cat_tensor(list_of_tensors, dim=-1):
    list_of_tensors = [t for t in list_of_tensors if t is not None]
    if len(list_of_tensors) > 1:
        result = torch.cat(list_of_tensors, dim=dim)
    elif len(list_of_tensors) == 1:
        result = list_of_tensors[0]
    else:
        result = None
    return result

