ansor='../tvm-ansor/'
codediffusion='../tvm-bayesian_code_diffusion/'

ansor_run='ansor.py'
codediffusion_run='our.py'

num_measures_per_round=64
num_trials=200
i=0
gpu_num=0

for model in resnet-18 mobilenet mobilenetv2 squeezenet_v1.1 inception_v3 mxnet bert vgg-16 vgg-19 efficientnet; do
for target in llvm cuda; do

tag='ansor'
log_dir=log_$tag; mkdir $log_dir 
dir_name=$log_dir/$model; mkdir $dir_name
dir_name=$log_dir/$model/$i; mkdir $dir_name
(
    export TVM_HOME=$ansor 
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} 
    CUDA_VISIBLE_DEVICES=$gpu_num python3 $ansor_run --target=$target --model=$model\
        --log_dir=$log_dir\
        --num_measures_per_round=$num_measures_per_round --test_idx=$i --num_trials=$num_trials > $dir_name/ansor-$model-$num_measures_per_round.out
)

tag='codediffusion-sketch'
log_dir=log_$tag; mkdir $log_dir 
dir_name=$log_dir/$model; mkdir $dir_name
dir_name=$log_dir/$model/$i; mkdir $dir_name
(
    export TVM_HOME=$codediffusion 
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} 
    CUDA_VISIBLE_DEVICES=$gpu_num python3 $codediffusion_run --target=$target --model=$model\
        --log_dir=$log_dir\
        --group_type=sketch\
        --num_measures_per_round=$num_measures_per_round --test_idx=$i --num_trials=$num_trials > $dir_name/codediffusion-$model-$num_measures_per_round.out
)

tag='codediffusion-operator'
log_dir=log_$tag; mkdir $log_dir 
dir_name=$log_dir/$model; mkdir $dir_name
dir_name=$log_dir/$model/$i; mkdir $dir_name
(
    export TVM_HOME=$codediffusion 
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} 
    CUDA_VISIBLE_DEVICES=$gpu_num python3 $codediffusion_run --target=$target --model=$model\
        --log_dir=$log_dir\
        --group_type=operator\
        --num_measures_per_round=$num_measures_per_round --test_idx=$i --num_trials=$num_trials > $dir_name/codediffusion-$model-$num_measures_per_round.out
)
done
done