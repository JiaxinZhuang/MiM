##########
# Server 6
source ./default_scripts/gpu6.sh
source ./default_scripts/default.sh
source ./default_scripts/mim.sh
exp=`echo ${0%.sh}`
exp=`echo ${exp#*/}`
output_dir=$prefix_dir/$exp

#################### TODO START
CUDA_VISIBLE_DEVICES=${mapping[4]}
fold=0
dataset_name=BTCV
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`

pretrained_path=$PATH_TO_PRETRAINED_WEIGHTS/checkpoint-final.pth
sr_ratio=1
batch_size=1
accum_iter=4
#################### TODO END

mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --rdzv-backend=c10d --max-restarts=3\
    --nproc-per-node=$NUM_TRAINERS --rdzv-endpoint=localhost:$port\
    main_finetune_segmentation.py \
    --fold $fold\
    --model_name convit3d\
    --dataset_name $dataset_name\
    --pretrained_path $pretrained_path\
    --sr_ratio $sr_ratio\
    --batch_size $batch_size\
    --accum_iter $accum_iter\
    --logdir $output_dir 2>&1 | tee $output_dir/log.txt