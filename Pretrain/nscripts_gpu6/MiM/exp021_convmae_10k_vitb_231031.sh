####################
# Server 6
source ./default_scripts/gpu6.sh
source ./default_scripts/default.sh
####################

#################### TODO START
CUDA_VISIBLE_DEVICES=${mapping[3]},${mapping[2]} #TODO
exp=`echo ${0%.sh}`
exp=`echo ${exp#*/}`
logdir=$prefix_dir/$exp
num_workers=16
source ./default_scripts/default.sh

dataset_split='+10k'
blr=1e-4
save_fq=10
batch_size=4
accum_iter=8
#dataset_loader=v1
dataset_loader=mmsmae
data_path=/home/jiaxin/data 
cache_dir=/data/jiaxin/cache/mmsmae_10k_231031
sr_ratio=1

mkdir -p $logdir
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
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
    --not_use_attn\
    --logdir $logdir 2>&1 | tee -a $logdir/train.log