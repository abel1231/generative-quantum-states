source ~/.bashrc
conda activate gps

# 获取输入的GPU ID
gpu_ids=$1

# 如果没有指定GPU ID，使用所有可用的GPU
if [ -z "$gpu_ids" ]; then
  gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
fi

# 构建命令
cmd="CUDA_VISIBLE_DEVICES=$gpu_ids python -u heisenberg_train_transformer.py"

# 打印命令
echo "Running command: $cmd"

# 运行命令
eval $cmd