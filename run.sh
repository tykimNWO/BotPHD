#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source  ~/.bashrc
conda   activate   CML

ml purge
ml load cuda/11.0

# Tmall
# python ./main.py --dataset=Tmall --opt_base_lr=1e-3 --opt_max_lr=5e-3 --opt_weight_decay=1e-4 --meta_opt_base_lr=1e-4 --meta_opt_max_lr=2e-3 --meta_opt_weight_decay=1e-4 --meta_lr=1e-3 --batch=8192 --meta_batch=128
# python ./main.py --dataset=Tmall --SSL_batch=18
# python ./main.py --dataset=Tmall --opt_base_lr=1e-3 --opt_max_lr=5e-3 --opt_weight_decay=1e-4 --meta_opt_base_lr=1e-4 --meta_opt_max_lr=2e-3 --meta_opt_weight_decay=1e-4 --meta_lr=1e-3 --batch=8192 --meta_batch=128 --SSL_batch=18 --isload=True --isJustTest=True
# python ./main.py --dataset=Tmall --opt_base_lr=1e-3 --opt_max_lr=5e-3 --opt_weight_decay=1e-4 --meta_opt_base_lr=1e-4 --meta_opt_max_lr=2e-3 --meta_opt_weight_decay=1e-4 --meta_lr=1e-3 --batch=8192 --meta_batch=128 --SSL_batch=18
# python ./main.py --dataset=Tmall --opt_base_lr=1e-3 --opt_max_lr=5e-3 --opt_weight_decay=1e-4 --meta_opt_base_lr=1e-4 --meta_opt_max_lr=2e-3 --meta_opt_weight_decay=1e-4 --meta_lr=1e-3 --batch=8192 --meta_batch=128 --SSL_batch=18 --isload=True --epoch=1

# IJCAI
python ./main.py --dataset=IJCAI_15 --sampNum=10 --opt_base_lr=1e-3 --opt_max_lr=2e-3 --opt_weight_decay=1e-4 --meta_opt_base_lr=1e-4 --meta_opt_max_lr=1e-3 --meta_opt_weight_decay=1e-4 --meta_lr=1e-3 --batch=8192 --meta_batch=128 --SSL_batch=30 

# RetailRocket
#python ./main.py --dataset='retailrocket' --sampNum=40 --lr=3e-4 --opt_base_lr=1e-4 --opt_max_lr=1e-3 --opt_weight_decay=1e-4 --opt_weight_decay=1e-4 --meta_opt_base_lr=1e-4 --meta_opt_max_lr=1e-3 --meta_opt_weight_decay=1e-3 --meta_lr=1e-3 --batch=2048 --meta_batch=128 --SSL_batch=15

echo "###"
echo "### END DATE=$(date)"