import optuna

from pathlib import Path
storage_path = Path("/mnt/obs/ye_canming/boguan_yuequ/peft")
storage_path.mkdir(exist_ok=True)
storage_name = f"sqlite:///{storage_path / 'optuna_studies.db'}"

# 1. 加载你现有的 Study
# 确保 study_name 和 storage 的路径是正确的
study = optuna.load_study(
    study_name="peft-method_comparison-MetaMathQA-gsm8k-ablation64-formal", 
    storage=storage_name
)

# 2. 找出所有失败的 trial
failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

# 3. 将这些 trial 的参数重新添加到队列中
# Optuna 会在下一批次中优先运行队列里的 trial
print(f"找到了 {len(failed_trials)} 个失败的实验，正在将它们重新入队...")
for trial in failed_trials:
    # trial.params 包含了导致失败的那组参数
    study.enqueue_trial(trial.params)
    print(f"已将参数 {trial.params} 重新加入队列。")