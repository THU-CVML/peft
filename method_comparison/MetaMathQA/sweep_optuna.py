# Copyright 2025-present Ye Canming.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hyperparameter sweep utility using Optuna.

This script supports two modes:
1. `tune`: For efficient hyperparameter exploration using CMA-ES Sampler and Hyperband Pruner.
2. `ablation`: For reproducible ablation studies using Grid Search and no pruning.
"""

import argparse
import json
import operator
import shutil
import subprocess
import sys
import uuid
from functools import reduce
from pathlib import Path

import optuna
import yaml


def set_nested_key(d: dict, key_path: str, value):
    """Sets a value in a nested dictionary using a dot-separated path."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def objective(trial: optuna.Trial, base_experiment_path: str, optuna_config: dict) -> float:
    """
    The Optuna objective function.
    """
    # 1. Create a unique directory for this trial that conforms to the expected path structure.
    base_path = Path(base_experiment_path)
    trial_name = f"{base_path.name}-trial-{trial.number}-{uuid.uuid4().hex[:8]}"
    trial_path = base_path.parent.parent / "optuna_sweeps" / trial_name

    print(f"\n--- Starting Trial {trial.number} ---\nPath: {trial_path}")

    # Copy the template directory to the new trial path.
    shutil.copytree(
        base_experiment_path,
        trial_path,
        ignore=shutil.ignore_patterns("optuna_sweep", "results.json", "*.log", "__pycache__")
    )

    # 2. Suggest hyperparameters and modify config files
    for config_filename, params in optuna_config.get('files', {}).items():
        config_path = trial_path / config_filename
        if not config_path.exists():
            print(f"[Trial {trial.number}] Warning: Config file '{config_filename}' not found. Skipping.")
            continue

        filetype = config_filename.split('.')[-1]

        with open(config_path, 'r') as f:
            if filetype == "json":
                config_data = json.load(f)
            elif filetype == "yaml":
                config_data = yaml.safe_load(f)
            else:
                raise NotImplementedError(f"Unsupported config file type: {filetype}")

        for key_path, suggest_config in params.items():
            suggest_type = suggest_config['type']
            suggest_args = suggest_config['args']
            
            suggest_method = getattr(trial, f"suggest_{suggest_type}")
            value = suggest_method(**suggest_args)
            
            set_nested_key(config_data, key_path, value)
            print(f"[Trial {trial.number}] Set {key_path} = {value}")

        with open(config_path, 'w') as f:
            if filetype == "json":
                json.dump(config_data, f, indent=4)
            elif filetype == "yaml":
                yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
            
    # 3. Launch run.py as a subprocess and capture its output
    log_path = trial_path / f"trial_{trial.number}.log"
    command = [sys.executable, "run.py", str(trial_path), "--verbose"]
    # command = [sys.executable, "run_accelerate.py", str(trial_path), "--verbose"]
    # command = ["accelerate", "launch", "--config_file", "./accelerate.yaml", "--num_processes", "1", "run_accelerate.py", str(trial_path), "--verbose"]

    print(f"[Trial {trial.number}] Executing command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True, # Decode stdout/stderr as text
        encoding='utf-8'
    )

    # 4. Monitor the output in real-time
    final_value = None
    intermediate_metric_key = optuna_config['intermediate_metric']
    final_metric_key = optuna_config['final_metric']
    print(f"[Trial {trial.number}] Monitoring output for intermediate metric '{intermediate_metric_key}' and final metric '{final_metric_key}'")

    # Open log file to save the output
    with open(log_path, 'w') as log_file:
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            log_file.write(line)
            
            try:
                log_data = json.loads(line)
                
                if intermediate_metric_key in log_data:
                    step = log_data.get("step")
                    metric_value = log_data[intermediate_metric_key]
                    trial.report(metric_value, step)
                    print(f"[Trial {trial.number}] Reported intermediate metric: Step {step}, {intermediate_metric_key}: {metric_value}")
                    
                    if trial.should_prune():
                        print(f"[Trial {trial.number}] Pruning trial.")
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise optuna.TrialPruned()

                if final_metric_key in log_data:
                     final_value = log_data[final_metric_key]
                     print(f"[Trial {trial.number}] Found final metric: {final_value}")

            except (json.JSONDecodeError, TypeError):
                continue
    
    process.wait()

    print(f"[Trial {trial.number}] Finished with exit code: {process.returncode}")

    if final_value is None:
        print(f"[Trial {trial.number}] Could not find final metric '{final_metric_key}' in log. Assuming failure.")
        raise RuntimeError("Final metric could not be determined.")
        
    return final_value


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps with Optuna.")
    parser.add_argument(
        "path_experiment", 
        type=str, 
        help="Path to the base experiment directory containing configs (train_params.json, optuna_config.yaml, etc.)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tune",
        choices=["tune", "ablation"],
        help="Set the operation mode: 'tune' for hyperparameter search, 'ablation' for grid search."
    )
    args = parser.parse_args()

    # --- Load Optuna Configuration ---
    optuna_config_path = Path(args.path_experiment) / "optuna_config.yaml"
    if not optuna_config_path.exists():
        print(f"Error: `optuna_config.yaml` not found in '{args.path_experiment}'")
        sys.exit(1)
        
    with open(optuna_config_path, 'r') as f:
        optuna_config = yaml.safe_load(f)
        
    print("Loaded Optuna configuration:")
    print(yaml.dump(optuna_config, indent=2))

    # --- Setup Optuna Study ---
    study_name = optuna_config.get("study_name", "peft_sweep")
    n_trials = optuna_config.get("n_trials", 20)
    direction = optuna_config.get("direction", "maximize")

    # storage_path = Path("/mnt/obs/ye_canming/boguan_yuequ/peft")
    storage_path = Path(".")
    storage_path.mkdir(exist_ok=True)
    storage_name = f"sqlite:///{storage_path / 'optuna_studies.db'}"

    # --- Configure Sampler and Pruner based on mode ---
    if args.mode == 'ablation':
        print("\n[Mode: Ablation] Using Grid Search Sampler and disabling pruning for reproducibility.")
        search_space = {}
        for _, params in optuna_config.get('files', {}).items():
            for _, suggest_config in params.items():
                if suggest_config['type'] != 'categorical':
                    raise ValueError(f"In 'ablation' mode, all hyperparameters must be of type 'categorical'.")
                param_name = suggest_config['args']['name']
                choices = suggest_config['args']['choices']
                search_space[param_name] = choices
        
        sampler = optuna.samplers.GridSampler(search_space)
        pruner = optuna.pruners.NopPruner()
        
        grid_size = reduce(operator.mul, (len(v) for v in search_space.values()), 1) if search_space else 1
        if n_trials != grid_size:
            print(f"Warning: In Grid Search mode, n_trials ({n_trials}) does not match the grid size ({grid_size}). Adjusting n_trials to {grid_size}.")
            n_trials = grid_size
    
    elif args.mode == 'tune':
        print("\n[Mode: Tune] Using CMA-ES Sampler and Hyperband Pruner for efficient tuning.")
        sampler = optuna.samplers.CmaEsSampler(consider_pruned_trials=True)
        pruner = optuna.pruners.HyperbandPruner()
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction=direction,
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    # --- Run Optimization ---
    try:
        study.optimize(
            lambda trial: objective(trial, args.path_experiment, optuna_config),
            n_trials=n_trials
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred during the optimization: {e}")
        print("The study will now conclude. You may be able to resume it later.")


    # --- Print Results ---
    print("\n--- Sweep Complete ---")
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    failed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Failed trials: {len(failed_trials)}")
    print(f"  Complete trials: {len(complete_trials)}")

    if study.best_trial:
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("No successful trials were completed.")


if __name__ == "__main__":
    main()
