# python enqueue_optuna.py
# python sweep_optuna.py experiments/yuequ-ark/latent8-ablation --mode ablation
# python sweep_optuna.py experiments/yuequ-ark/latent32-tune --mode tune
export SWANLAB_PROJECT_NAME="Qwen2.5-3B-ARK-Baselines"

# 6
python run.py experiments/hf-peft-32-Qwen2.5-3B/adalora-32
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
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/ln_tuning
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/lokr
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/oft-r32
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/randlora-r32
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/vblora-r4
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/vera-r256


# 6
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/adaptionprompt
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/prefixtuning
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/prefixtuning-1e-3
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/prompt_tuning
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/prompt_tuning-1e-3
python run.py experiments/hf-peft-not-for-bert-Qwen2.5-3B/ptuning

