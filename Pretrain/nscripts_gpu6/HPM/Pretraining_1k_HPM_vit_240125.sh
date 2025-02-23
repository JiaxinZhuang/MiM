####################
# Server 6
source ./default_scripts/gpu6.sh
source ./default_scripts/default.sh
export prefix_dir=/home/jiaxin/ckpts/baselines
exp=`echo ${0%.sh}`
exp=`echo ${exp#*/}`
logdir=$prefix_dir/$exp
####################

#################### TODO START
CUDA_VISIBLE_DEVICES=${mapping[2]},${mapping[3]} #TODO
model_name=HPM_mae_vit_base_patch16
dataset_loader=HPM
dataset_split='1k'
num_workers=8
data_path=/home/jiaxin/data
cache_dir=/data/cache/cache_20k_20230825
batch_size=16
accum_iter=2
blr=1.5e-4
save_fq=50
mask_ratio=0.6

####################
mkdir -p $logdir
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --rdzv-backend=c10d --max-restarts=0\
    --nproc-per-node=$NUM_TRAINERS --rdzv-endpoint=localhost:$port main_pretrain.py \
    --model_name $model_name\
    --learning_loss \
    --relative \
    --token_size 6\
    --dataset_loader $dataset_loader\
    --dataset_split $dataset_split\
    --batch_size $batch_size \
    --norm_pix_loss \
    --mask_ratio $mask_ratio \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr $blr \
    --weight_decay 0.05 \
    --sw_batch_size 4\
    --num_workers $num_workers\
    --accum_iter $accum_iter\
    --data_path $data_path\
    --save_fq $save_fq\
    --cache_dir $cache_dir\
    --logdir $logdir 2>&1 | tee -a $logdir/train.log