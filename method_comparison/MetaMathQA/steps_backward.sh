#!/bin/bash

# Wait for CUDA device 0 to have enough free memory
wait_for_cuda_memory() {
    local device_id=3
    local min_free_mb=28000  # Minimum free memory in MB
    
    while true; do
        # Get free memory for device 0
        free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$device_id)
        
        if [ "$free_memory" -gt "$min_free_mb" ]; then
            echo "CUDA device $device_id has $free_memory MB free memory. Proceeding..."
            break
        else
            echo "CUDA device $device_id only has $free_memory MB free memory. Waiting..."
            sleep 10
        fi
    done
}

# Wait for sufficient GPU memory
wait_for_cuda_memory

# Run your next command here
echo "Running next command..."
# YOUR_COMMAND_HERE

export SWANLAB_PROJECT_NAME="Qwen2.5-3B-ARK-Baselines"
export CUDA_VISIBLE_DEVICES=3


# 6
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/ptuning
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/prompt_tuning-1e-3
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/prompt_tuning
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/prefixtuning-1e-3
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/prefixtuning
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/adaptionprompt


python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/vera-r256
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/vblora-r4
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/randlora-r32
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/oft-r32
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/lokr
# python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/ln_tuning # 跑过了






