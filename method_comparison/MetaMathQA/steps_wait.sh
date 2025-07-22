#!/bin/bash

# Wait for CUDA device 0 to have enough free memory
wait_for_cuda_memory() {
    local device_id=1
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

# echo test
python sweep_optuna.py experiments/yuequ-ark/latent32-ablation --mode ablation