source ./default_scripts/gpu6.sh
source ./default_scripts/default.sh

# CUDA_VISIBLE_DEVICES=${mapping[2]},${mapping[6]} #TODO
CUDA_VISIBLE_DEVICES=${mapping[6]} #TODO
# model_name=simMIM_swin
model_name=Adam
# model_name=MiM
# model_name=HPM_mae_vit_base_patch16
# model_name=localMIM_vit_base_patch16
logdir=/tmp/$model_name
mkdir $logdir

NUM_TRAINERS=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --rdzv-backend=c10d --max-restarts=0\
    --nproc-per-node=$NUM_TRAINERS --rdzv-endpoint=localhost:$port test_flops_params.py \
    --model_name $model_name\
    --logdir $logdir