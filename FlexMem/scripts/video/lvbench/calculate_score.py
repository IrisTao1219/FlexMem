import torch
import transformers
import json
import pandas as pd
from glob import glob
from decord import VideoReader, cpu

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--eval_type", default="multi_choice", help="The path to save annotation final combined json file.")
    parser.add_argument("--num-chunks", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    # wups, multi_choice
    pred_type=args.eval_type
    ori_dir = args.output_dir
    files = glob(ori_dir + f'/{args.num_chunks}_*')
    if len(files) == 0:
        files = glob(ori_dir + f'/*')
    # files = glob(ori_dir + f'/res.json')
    print(files)
    
    if pred_type == 'multi_choice':
        total_cnt, total_acc = 0, 0
        short_cnt, medium_cnt, long_cnt = 0, 0, 0
        short_acc, medium_acc, long_acc = 0, 0, 0
        for file in files:
            lines = open(file, 'r').readlines()
            for line in lines:
                line = eval(line)
                total_cnt += 1
                duration_group = line['duration_group']
                if duration_group == 15 or duration_group == 60:
                    short_cnt += 1
                elif duration_group == 600:
                    medium_cnt += 1
                elif duration_group == 3600:
                    long_cnt += 1

                if line['acc'] == 'True':
                    total_acc += 1
                    if duration_group == 15 or duration_group == 60:
                        short_acc += 1
                    elif duration_group == 600:
                        medium_acc += 1
                    elif duration_group == 3600:
                        long_acc += 1
                        
        assert short_cnt + medium_cnt + long_cnt == total_cnt
        if short_cnt > 0:
            print(f'short acc: {short_acc / short_cnt}')
            print(f'short_cnt: {short_cnt}')
        if medium_cnt > 0:
            print(f'medium acc: {medium_acc / medium_cnt}')
            print(f'medium_cnt: {medium_cnt}')
        if long_cnt > 0:
            print(f'long acc: {long_acc / long_cnt}')
            print(f'long_cnt: {long_cnt}')
        print(f'total acc: {total_acc / total_cnt}, total_cnt: {total_cnt}')
        

if __name__ == "__main__":
    main()