# %%
# 读取 deepspeed
import inspect
from swift.llm.argument.train_args import TrainArguments
import os

# TARGET_MODE = "zero2"
TARGET_MODE = "zero2_offload"
SUPPORTED_MODES = [
    'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'
]
OUTPUT_FILE = "deepspeed.json"

# 通过inspect获得train_args.py的路径
train_args_file = inspect.getfile(TrainArguments)
ds_config_folder = os.path.abspath(os.path.join(os.path.dirname(train_args_file), '..', 'ds_config'))

deepspeed_mapping = {
    name: f'{name}.json'
    for name in SUPPORTED_MODES
}

deepspeed_config = deepspeed_mapping[TARGET_MODE]
deepspeed_config = os.path.join(ds_config_folder, deepspeed_config)
print(f"DeepSpeed config path: {deepspeed_config}")
# 直接复制
import shutil
shutil.copy(deepspeed_config, OUTPUT_FILE)
print(f"Copied {deepspeed_config} to {OUTPUT_FILE}")

# import json
# with open(deepspeed_config, 'r', encoding='utf-8') as f:
#     deepspeed_config = json.load(f)
# with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#     json.dump(deepspeed_config, f, indent=4, ensure_ascii=False)
# %%
