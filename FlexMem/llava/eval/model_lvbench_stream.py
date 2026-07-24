import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math

import random
import numpy as np
from glob import glob
from decord import VideoReader, cpu
import sys
import yaml
import warnings
warnings.filterwarnings("ignore")

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        # pred_index = random.choice(all_choices)
        return ''
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def eval_model(args):
    
    #get config
    configs = load_yaml(args.config_path)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    if args.native:
        model_name = get_model_name_from_path(model_path)
    else:
        model_name = get_model_name_from_path(model_path) + '_stream'
    overwrite_config = {}
    if configs["tokens_per_frame"] == 182:
        overwrite_config["mm_spatial_pool_stride"] = 2
        overwrite_config["mm_spatial_pool_mode"] = "average"
        overwrite_config["mm_pooling_position"] = "before"
    overwrite_config["tokens_per_frame"] = configs["tokens_per_frame"]
    overwrite_config["mm_newline_position"] = "grid"
    overwrite_config["delay_load"] = False
    if args.native:
        overwrite_config["_attn_implementation"] = args.attn_implementation
    else:
        overwrite_config["_attn_implementation"] = 'eager'
    topk , chunk_size , preblk , preratio , decratio , topb, max_fps = configs["max_num_frames"],configs["chunk_size"],configs["preblk"],configs["preratio"],configs["decratio"],configs["topb"],configs["sample_fps"]
    if args.native:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            args.model_base,
            model_name,
            overwrite_config=overwrite_config,
            attn_implementation=args.attn_implementation,
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, overwrite_config=overwrite_config, attn_implementation="eager" , chunk_size = chunk_size , topb = topb , preblk = preblk , preratio= preratio,decratio = decratio, tokens_per_frame = configs["tokens_per_frame"])
    model = model.eval()

    # Data
    data = json.load(open(args.gt_file, 'r'))
    gt_questions = []
    for item in data:
        question = item['question']
        option = [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(item["candidates"])]
        qid = item['id']
        video_id = item["video_path"].split('.')[0]
        answer_id = chr(ord("A")+item["correct_choice"])
        answer = item["candidates"][item["correct_choice"]]
        duration_group = item['duration_group']
        index2ans = {}
        for i in range(len(item["candidates"])):
            idx = chr(ord("A")+i)
            ans = item["candidates"][i]
            index2ans[idx] = ans
        gt_questions.append({
            'qid': qid, 
            'question': question, 
            'option': option, 
            'video_id': video_id, 
            'answer_id': answer_id, 
            'answer': answer, 
            'index2ans': index2ans,
            'duration_group': duration_group,
        })

    # random.seed(42)
    # random.shuffle(gt_questions)
    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"{args.num_chunks}_{args.chunk_idx}" if args.num_chunks > 1 else args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.jsonl")

    questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        qid = line["qid"]
        answer = line["answer"]
        video_id = line["video_id"]
        answer_id = line["answer_id"]
        option = line["option"]
        index2ans = line["index2ans"]
        duration_group = line['duration_group']

        ori_question = line['question']
        question = [line['question']] + option
        question = '\n'.join(question)
        question_w_options = question
        question = f'{question}\nPlease answer directly with only the letter of the correct option and nothing else.'

        sample_set = {
            "id": qid, 
            "video_id": video_id,
            "question": question, 
            "answer": answer, 
            "answer_id": answer_id, 
            'duration_group': duration_group,
        }

        video_path = os.path.join(args.video_dir, f'{video_id}.mp4')

        # Check if the video exists
        if not os.path.exists(video_path):
            print(f'Miss video {video_id}')
            continue

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = vr.get_avg_fps()
        video_time = total_frame_num / fps

        if args.native:
            nframes = args.num_frames
            if nframes <= 0:
                raise ValueError("--num-frames must be a positive integer.")
        else:
            fps_sample_frames = int(video_time*max_fps)
            nframes = min(fps_sample_frames, topk)
            if nframes % chunk_size != 0:
                nframes += (chunk_size - nframes % chunk_size)
        
        uniform_sampled_frames = np.linspace(0, total_frame_num-1, nframes, dtype=int)

        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/fps for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        video = vr.get_batch(frame_idx).asnumpy()
        video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
        video = [video]
        qs = question
        time_instruciton = ''
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + time_instruciton + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + time_instruciton + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        if args.native:
            estimated_context_tokens = (
                input_ids.shape[1]
                - 1
                + nframes * configs["tokens_per_frame"]
                + args.max_new_tokens
            )
            if estimated_context_tokens > context_len:
                raise RuntimeError(
                    f"Native context overflow for video {video_id}: estimated "
                    f"{estimated_context_tokens} tokens exceed context_len={context_len}. "
                    "Reduce --num-frames."
                )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        question_ids = tokenizer(question_w_options, return_tensors="pt").input_ids
        qs_len = question_ids.shape[1]

        if args.generate_method == 'generate_until':

            if args.native:
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=video,
                        modalities=["video"],
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=1,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                    )
            else:
                model.memory.reset()
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        question_ids=question_ids,
                        is_parallel=False,
                        qid=qid,

                        images=video,
                        modalities= ["video"],
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=1,
                        max_new_tokens=1024,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria]
                    )

        
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            if args.native:
                if 'assistant\n' in outputs:
                    outputs = outputs.rsplit('assistant\n', 1)[-1].strip()
            else:
                outputs = outputs.split('assistant\n')[1]

            parsed_pred = parse_multi_choice_response(outputs, ["A", "B", "C", "D", "E"], index2ans)
            sample_set['acc'] = str(parsed_pred == answer_id)   
            sample_set["pred"] = outputs
            if args.native:
                sample_set["method"] = "native"
                sample_set["num_frames"] = nframes
                sample_set["estimated_visual_tokens"] = nframes * configs["tokens_per_frame"]



        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()


    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--gt_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--output_dir", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--config_path", type=str, default='flat')
    parser.add_argument("--generate_method", type=str, default='generate_until')
    parser.add_argument("--for_get_frames_num", type=int, default=99)
    parser.add_argument("--native", action="store_true", help="Use native LLaVA-Qwen inference without FlexMem.")
    parser.add_argument("--num-frames", type=int, default=64, help="Uniformly sample this many frames in native mode.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Maximum generated tokens in native mode.")
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"], help="Attention backend used only in native mode.")
    args = parser.parse_args()

    eval_model(args)