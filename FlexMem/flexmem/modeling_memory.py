
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
import torch.nn.functional as F

logger = logging.get_logger(__name__)


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
        self.start_idx = 0
        self.end_idx = 0
        self.all_beacon_sizes = []
        self.batch_loss = None
        self.valid_token_num = None
        self.step_idx = 0
        self.compression_ratio = None
        self.is_full_window = True
        self.raw_size_to_cache = 0
        self.qs_len = 0
        self.question_ids = None
        self.is_decoding = False
        self.interleave_remainder = 0
        self.interleave_compression_ratio = None
        self.beacon_indices = None
        self.all_input_ids = None
        self.all_attention_mask = None
        self.all_labels = None
        self.beacon_skip_first = None
        self.beacon_skip_last = None
        self.raw_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        self.sink_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        self.beacon_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        self.visual_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        self.compression_activations = [([],[]) for _ in range(self.config.num_hidden_layers)]
        self.long_term_memory = [(None, None) for _ in range(self.config.num_hidden_layers)]

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
        if self.all_input_ids is None:
            self.all_input_ids = input_ids.cpu() # for prefill
        else:
            self.all_input_ids = torch.cat([self.all_input_ids, input_ids.cpu()], dim=1) # for decoding
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=torch.device("cpu"))
        if self.all_attention_mask is None:
            self.all_attention_mask = attention_mask.cpu()
        else:
            self.all_attention_mask = torch.cat([self.all_attention_mask, attention_mask.cpu()], dim=1)
        if labels is not None:
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

        clip_num = int(imgs_len / self.config.beacon_window)
        self.all_selction_scores = torch.zeros((clip_num,), device=self._device, dtype=torch.float32)


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
                    assert _past_key_values[0][0] is None
            
            batch_size = self.all_input_ids.shape[0]
            seq_len = sum(x.shape[1] for x in input_ids_list) + sum(beacon_size_list) - beacon_size_list[-1]

            input_ids = _input_ids.new_zeros((batch_size, seq_len)) + self.beacon_token
            attention_mask = _attention_mask.new_zeros((batch_size, 1, seq_len, mem_size + seq_len)) + self.min_value
            position_ids = torch.arange(mem_size + seq_len, device=self._device).expand(batch_size, mem_size + seq_len)
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

                # populate input_ids
                _input_ids = input_ids_list[i]
                cur_seq_len = _input_ids.shape[1]
                input_ids[:, start_idx: start_idx + cur_seq_len] = _input_ids
                
                # populate attention_mask and position_ids
                _attention_mask = attention_mask_list[i]
                _position_ids = position_ids_list[i]
                # the attention mask in the first window contains the mask for memory, which is redundant here
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
        start_idx = self.start_idx
        end_idx = start_idx + self.beacon_window
        if end_idx > self.all_sequence_length:
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

        if self.all_labels is not None:
            labels = self.all_labels[:, self.end_idx: end_idx].to(self._device)
        else:
            labels = None
        batch_size = input_ids.shape[0]

        #============================================#
        # Insert beacon tokens if necessary.
        #============================================#

        beacon_indices = self.qs_len
        #============================================#
        past_key_values = []
        for layer_idx in range(self.config.num_hidden_layers):
            if ignore_memory:
                key, value = None, None
            else:
                sink_key, sink_value = self.sink_activations[layer_idx]
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

        for layer_idx, (key, value, beacon_size, beacon_indices, selection_score, long_term_info) in enumerate(past_key_values):
            # NOTE: the past_key_values are incrementally returned (only the new keys and values are returned)
            previous_raw_key, previous_raw_value = self.raw_activations[layer_idx]
            previous_compress_key, previous_compress_value = self.compression_activations[layer_idx]
            previous_visual_key, previous_visual_value = self.visual_activations[layer_idx]


            if selection_score is not None and layer_idx > 2: # 
                self.all_selction_scores[step_num-2] += selection_score

            # sink activations is used to store the key and value of system prompts
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

                # this means the current input does not fulfill a window
                # thus, the key and value are all raw activations, and we accumulate them until the window is fulfilled
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

                # NOTE: select topk key segments relevant to query
                # if beacon_size == -100 and key.shape[2] != 1:
                #     self.beacon_activations[layer_idx] = (None, None)
                #     continue

            else:
                # NOTE: use the correct previous_beacon_key and value!
                previous_beacon_key, previous_beacon_value = self.beacon_activations[layer_idx]
                previous_long_term_key, previous_long_term_value = self.long_term_memory[layer_idx]
               
                beacon_key, beacon_value = key, value
                visual_key, visual_value = key, value
                
                long_term_key, long_term_value = long_term_info

                long_term_key = cat_tensor([
                    previous_long_term_key, 
                    long_term_key
                ], dim=self.k_seq_dim)
                long_term_value = cat_tensor([
                    previous_long_term_value, 
                    long_term_value
                ], dim=self.v_seq_dim)
                visual_size = int(self.preblk*long_term_info[0].shape[self.k_seq_dim]) #
                if long_term_key.shape[self.k_seq_dim] > visual_size:
                    long_term_key = slice_tensor(long_term_key, start=-visual_size, dim=self.k_seq_dim)
                    long_term_value = slice_tensor(long_term_value, start=-visual_size, dim=self.v_seq_dim)

                if step_num > self.preblk+1: 
                    recent_blk_num = int(self.preblk * 0.5)
                    recent_visual_size = int(recent_blk_num*long_term_info[0].shape[self.k_seq_dim])
                    recent_key, recent_value = slice_tensor(long_term_key, start=-recent_visual_size, dim=self.k_seq_dim), slice_tensor(long_term_value, start=-recent_visual_size, dim=self.v_seq_dim)
                    
                    bsz, Hkv, chunk_size, D = visual_key.shape
                    long_blk_num = int((visual_size - recent_visual_size)/chunk_size)
                    candidates_blk_num = step_num-1-recent_blk_num
                    selection_scores = self.all_selction_scores[:candidates_blk_num]
                    _, keep_indices = selection_scores.topk(min(long_blk_num, len(selection_scores)))
                    keep_indices = keep_indices.sort().values
                    long_key, long_value = previous_visual_key.index_select(dim=2, index=keep_indices).reshape(bsz, Hkv, -1, D), previous_visual_value.index_select(dim=2, index=keep_indices).reshape(bsz, Hkv, -1, D)
                    beacon_key = cat_tensor([long_key, recent_key], dim=self.k_seq_dim)
                    beacon_value = cat_tensor([long_value, recent_value], dim=self.v_seq_dim)
                else:
                    beacon_key, beacon_value = long_term_key, long_term_value

                visual_key = cat_tensor([
                    previous_visual_key, 
                    visual_key.unsqueeze(2)
                ], dim=2)
                visual_value = cat_tensor([
                    previous_visual_value, 
                    visual_value.unsqueeze(2)
                ], dim=2)

                self.beacon_activations[layer_idx] = (beacon_key, beacon_value)
                self.visual_activations[layer_idx] = (visual_key, visual_value)
                self.long_term_memory[layer_idx] = (long_term_key, long_term_value)

        if self.start_idx == self.beacon_skip_last:
            selection_scores = self.all_selction_scores
            retrieval_clip_num = min(self.config.topk_clips, len(selection_scores))                                   
            _, keep_indices = selection_scores.topk(retrieval_clip_num)
            keep_indices = keep_indices.sort().values 
        
            for layer_idx, (key, value, beacon_size, beacon_indices, selection_score, long_term_info) in enumerate(past_key_values):
                visual_key, visual_value = self.visual_activations[layer_idx]
                bsz, head_nums = visual_key.shape[:2]
                dim = visual_key.shape[-1]
      
                if self.config.topk_clips < len(selection_scores):
                    visual_key = visual_key.index_select(dim=2, index=keep_indices)
                    visual_value = visual_value.index_select(dim=2, index=keep_indices)
                
                visual_key = visual_key.reshape(bsz, head_nums, -1, dim)
                visual_value = visual_value.reshape(bsz, head_nums, -1, dim)
                beacon_key, beacon_value = visual_key, visual_value 
                self.beacon_activations[layer_idx] = (beacon_key, beacon_value)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

