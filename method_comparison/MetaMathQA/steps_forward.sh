# python enqueue_optuna.py
# python sweep_optuna.py experiments/yuequ-ark/latent8-ablation --mode ablation
# python sweep_optuna.py experiments/yuequ-ark/latent32-tune --mode tune
export SWANLAB_PROJECT_NAME="Qwen2.5-3B-ARK-Baselines"

# 6
# python run.py experiments/hf-peft-32-Qwen2.5-3B/adalora-32 # 跑过了
python run.py experiments/hf-peft-32-Qwen2.5-3B/bone-32
python run.py experiments/hf-peft-32-Qwen2.5-3B/bone-bat-32
python run.py experiments/hf-peft-32-Qwen2.5-3B/loha-32
python run.py experiments/hf-peft-32-Qwen2.5-3B/lora-32
python run.py experiments/hf-peft-32-Qwen2.5-3B/lora-dora-32

# 10
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/boft-b4
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/fourierft-f1000
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/fourierft-f5000
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/ia3
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/ia3-1e-3












