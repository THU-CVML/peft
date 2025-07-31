# python enqueue_optuna.py
# python sweep_optuna.py experiments/yuequ-ark/latent8-ablation --mode ablation
# python sweep_optuna.py experiments/yuequ-ark/latent32-tune --mode tune
export SWANLAB_PROJECT_NAME="Qwen2.5-3B-ARK-Baselines"
python run.py experiments/hf-peft-not-scaled-Qwen2.5-3B/ln_tuning