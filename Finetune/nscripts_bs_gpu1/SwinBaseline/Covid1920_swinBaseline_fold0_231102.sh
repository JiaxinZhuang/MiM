##########
# Server 1
source ./default_scripts/gpu1.sh
source ./default_scripts/default.sh
exp=`echo ${0%.sh}`
exp=`echo ${exp#*/}`
output_dir=$prefix_dir/$exp
##########

#################### TODO START
CUDA_VISIBLE_DEVICES=${mapping[5]}
NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
dataset_name=Covid19_20
fold=0
roi_x=192
roi_y=192
roi_z=16
optim_lr=1e-4
batch_size=1
accum_iter=2
#################### TODO END

mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u main_finetune_segmentation.py \
    --model_name swin_unetr\
    --fold $fold\
    --dataset_name $dataset_name\
    --optim_lr $optim_lr\
    --batch_size $batch_size\
    --logdir $output_dir 2>&1 | tee -a $output_dir/log.txt