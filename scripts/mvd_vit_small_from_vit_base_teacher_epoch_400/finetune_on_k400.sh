#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`
OUTPUT_DIR='OUTPUT/mvd_vit_small_with_vit_base_teacher_k400_epoch_400/finetune_on_k400'
MODEL_PATH='/data/i5O/finetuned/july24/checkpoint-24/mp_rank_00_model_states.pt'
DATA_PATH='/data/i5O/kinetics-dataset/annotations'
DATA_ROOT='/data/i5O/kinetics400/train/'

MASTER_ADDR=127.0.0.1
MASTER_PORT=6006
NODE_COUNT=1
RANK=0

# train on 16 V100 GPUs (2 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
    --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set Kinetics-400 --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size 3 --update_freq 2 --num_sample 2 \
    --save_ckpt_freq 1 \
	--start_epoch 25 \
    --num_frames 16 --sampling_rate 4 \
    --lr 1e-3 --warmup_lr 1e-3 --min_lr 1e-3 --warmup_epochs 0 --epochs 30 \
    --dist_eval --test_num_segment 5 --test_num_crop 3 \
    --enable_deepspeed \
    --num_workers 8
