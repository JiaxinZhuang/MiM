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
#CUDA_VISIBLE_DEVICES=${mapping[1]},${mapping[3]} #TODO
CUDA_VISIBLE_DEVICES=${mapping[1]} #TODO
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
dataset_loader=Adam
model_name=Adam
num_workers=16
data_path=/home/jiaxin/data
cache_dir=/data/jiaxin/cache/1k_mocov2_231106
dataset_split='1k'
blr=5e-3
save_fq=50
batch_size=64
accum_iter=1
moco_k=512
roi_x=96
roi_y=96
roi_z=96

granularity=0
#resume=/jhcnas1/jiaxin/ckpts/project_02/baselines/MoCoV2/Pretraining_1k_MoCov2_resnet_231106/checkpoint-final.pth
resume=/home/jiaxin/ckpts/baselines/Adam/Pretraining_1k_MoCov2_resnet_231106/checkpoint-final.pth
#################### TODO END

mkdir -p $logdir
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --rdzv-backend=c10d --max-restarts=0\
    --nproc-per-node=$NUM_TRAINERS --rdzv-endpoint=localhost:$port main_pretrain.py \
    --dataset_split $dataset_split\
    --dataset_loader $dataset_loader\
    --granularity $granularity\
    --batch_size $batch_size \
    --model_name $model_name\
    --epochs 300 \
    --warmup_epochs 40 \
    --blr $blr \
    --weight_decay 0.05 \
    --sw_batch_size 4\
    --num_workers $num_workers\
    --accum_iter $accum_iter\
    --data_path $data_path\
    --save_fq $save_fq\
    --cache_dir $cache_dir\
    --moco-k $moco_k\
    --resume $resume\
    --roi_x $roi_x\
    --roi_y $roi_y\
    --roi_z $roi_z\
    --logdir $logdir 2>&1 | tee -a $logdir/train.log