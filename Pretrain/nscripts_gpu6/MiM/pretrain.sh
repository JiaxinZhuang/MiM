####################
# Server 6
source ./default_scripts/gpu6.sh
source ./default_scripts/default.sh
#source ./default_scripts/mim.sh
exp=`echo ${0%.sh}`
exp=`echo ${exp#*/}`
logdir=$prefix_dir/$exp
####################

#################### TODO START
CUDA_VISIBLE_DEVICES=${mapping[4]},${mapping[1]} #TODO
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
num_workers=16
data_path=/home/jiaxin/data
cache_dir=/data/jiaxin/cache/mmsmae_1k_231004
dataset_split='1k'
sr_ratio=1
blr=1e-4
save_fq=10
batch_size=1
accum_iter=64
dataset_loader=mmsmae
atten_weight_uu=1e-1
atten_weight_ud=1e-1
reconstruct_weight_up=1.0
reconstruct_weight_usual=1.0
reconstruct_weight_down=1.0
#################### TODO END

mkdir -p $logdir
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --rdzv-backend=c10d --max-restarts=0\
    --nproc-per-node=$NUM_TRAINERS --rdzv-endpoint=localhost:$port main_pretrain.py \
    --batch_size $batch_size \
    --model_name convmae_convvit_base_patch16 \
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
    --attn_loss_name 'byol'\
    --atten_weight_uu $atten_weight_uu\
    --atten_weight_ud $atten_weight_ud\
    --reconstruct_weight_up $reconstruct_weight_up\
    --reconstruct_weight_usual $reconstruct_weight_usual\
    --reconstruct_weight_down $reconstruct_weight_down\
    --logdir $logdir 2>&1 | tee -a $logdir/train.log