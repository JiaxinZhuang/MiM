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
CUDA_VISIBLE_DEVICES=${mapping[4]},${mapping[5]} #TODO
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
num_workers=16
data_path=/home/jiaxin/data 
cache_dir=/data/jiaxin/cache/1k_mocov2_231106
dataset_split='1k'
blr=1e-3
save_fq=50
batch_size=32
accum_iter=4
dataset_loader=MoCoV2
model_name=MoCoV2
moco_k=512
max_epochs=800
#################### TODO END

mkdir -p $logdir
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --rdzv-backend=c10d --max-restarts=0\
    --nproc-per-node=$NUM_TRAINERS --rdzv-endpoint=localhost:$port main_pretrain.py \
    --dataset_split $dataset_split\
    --dataset_loader $dataset_loader\
    --batch_size $batch_size \
    --model_name $model_name\
    --warmup_epochs 40 \
    --blr $blr \
    --weight_decay 0.05 \
    --sw_batch_size 1\
    --num_workers $num_workers\
    --accum_iter $accum_iter\
    --data_path $data_path\
    --save_fq $save_fq\
    --cache_dir $cache_dir\
    --moco-k $moco_k\
    --epochs $max_epochs\
    --logdir $logdir 2>&1 | tee -a $logdir/train.log