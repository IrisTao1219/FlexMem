#!/bin/bash
ROOT_DIR="FlexMem"

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export DECORD_EOF_RETRY_MAX=20480
CKPT="/home/taosha/models/Hf_model/LLaVA-Video-7B-Qwen2" #Your Model Path
DATA_ROOT="/home/taosha/datasets/Datasets/LongVideoBench" #Your LongVideoBench Root
VIDEO_DIR=${DATA_ROOT}/videos
# GT_FILE=${DATA_ROOT}/lvb_val.json
GT_FILE="../bench/lvb_val_quick_600.json"
DURATION_GROUP=$(basename "$GT_FILE" | sed -E 's/.*_([0-9]+)\.json/\1/')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_NAME="pred_${DURATION_GROUP}_${TIMESTAMP}"
CONV_MODE=qwen_1_5
POOL_STRIDE=2
OVERWRITE=True
QUESTION_TYPE=multi_choice

EVAL_ONLY=False

# FRAMES=99 useless
CONFIG_PATH=config.yaml


GEN_METHOD=generate_until

OPENAIKEY="INPUT YOUR OPENAI API"

Test=0

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_test_${Test}_${GEN_METHOD}_${QUESTION_TYPE}_overwrite_${OVERWRITE}
else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_test_${Test}_${GEN_METHOD}_${QUESTION_TYPE}
fi

echo $SAVE_DIR

CHUNKS=1
# Assuming GPULIST is a bash array containing your GPUs
GPULIST=(1)

# Get the number of GPUs
NUM_GPUS=${#GPULIST[@]}

# Calculate GPUs per chunk
GPUS_PER_CHUNK=$((NUM_GPUS / CHUNKS))

echo "CKPT=[$CKPT]"
echo "DATA_ROOT=[$DATA_ROOT]"
echo "VIDEO_DIR=[$VIDEO_DIR]"
echo "GT_FILE=[$GT_FILE]"

if [ "$EVAL_ONLY" == False ]; then
    for IDX in $(seq 1 $CHUNKS); do
        START=$(((IDX-1) * GPUS_PER_CHUNK))
        LENGTH=$GPUS_PER_CHUNK # Length for slicing, not the end index
        CHUNK_GPUS=(${GPULIST[@]:$START:$LENGTH})
        CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")   
        echo "CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"
        CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR python3 llava/eval/model_lvbench_stream.py \
            --model-path $CKPT \
            --video_dir $VIDEO_DIR \
            --gt_file $GT_FILE \
            --output_dir ./work_dirs/eval_lvbench/$SAVE_DIR \
            --output_name "$OUTPUT_NAME" \
            --num-chunks $CHUNKS \
            --chunk-idx $(($IDX - 1)) \
            --config_path $CONFIG_PATH \
            --generate_method $GEN_METHOD \
            --conv-mode $CONV_MODE &
    done
            # --for_get_frames_num $FRAMES \ useless

    wait
fi

python3 ./scripts/video/lvbench/calculate_score.py \
    --output_dir ./work_dirs/eval_lvbench/$SAVE_DIR \
    --eval_type $QUESTION_TYPE \
    --num-chunks $CHUNKS \
    --output-name "$OUTPUT_NAME"
