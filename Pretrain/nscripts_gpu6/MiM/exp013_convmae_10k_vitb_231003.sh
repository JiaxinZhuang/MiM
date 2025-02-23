####################
# Server 6
declare -A mapping=( [0]=3 [1]=0 [2]=1 [3]=2 [4]=4 [5]=5 [6]=6 [7]=7) #TODO
prefix_dir=/home/jiaxin/ckpts/MMSMAE #TODO
# CUDA_VISIBLE_DEVICES=${mapping[6]},${mapping[7]} #TODO
CUDA_VISIBLE_DEVICES=${mapping[3]}} #TODO
####################
set -x
min_port=1024
max_port=65535
port=$((min_port + RANDOM % (max_port - min_port + 1)))
export OMP_NUM_THREADS=48
########## server
exp=`echo ${0%.sh}`
exp=`echo ${exp#*/}`
logdir=$prefix_dir/$exp
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
mkdir $logdir
########## experiments
num_workers=16
data_path=/home/jiaxin/data #TODO
dataset_split='+10k'
blr=5e-4
save_fq=10
batch_size=16
accum_iter=4
dataset_loader=v1
cache_dir=/data/cache/cache_20k_20230825
sr_ratio=1
####################
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --rdzv-backend=c10d --max-restarts=3\
    --nproc-per-node=$NUM_TRAINERS --rdzv-endpoint=localhost:$port main_pretrain.py \
    --batch_size $batch_size \
    --model convmae_convvit_base_patch16 \
    --norm_pix_loss \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr $blr \
    --weight_decay 0.05 \
    --sw_batch_size 4\
    --mask_ratio 0.6\
    --num_workers $num_workers\
    --accum_iter $accum_iter\
    --dataset_split $dataset_split\
    --data_path $data_path\
    --save_fq $save_fq\
    --cache_dir $cache_dir\
    --dataset_loader $dataset_loader\
    --persistent_dataset\
    --sr_ratio $sr_ratio\
    --logdir $logdir 2>&1 | tee -a $logdir/train.log