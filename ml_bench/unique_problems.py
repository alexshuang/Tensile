#!python3

import yaml
import os
import sys
from pathlib import Path

train_yaml = Path(sys.argv[1])
test_yaml = Path(sys.argv[2])
output_yaml = test_yaml.with_suffix('.diff')

os.system(f"sed -i '/^\s*$/d' {test_yaml}")
os.system(f'cp {test_yaml} {output_yaml} -v')

train_data = yaml.safe_load(open(train_yaml))
test_data = yaml.safe_load(open(test_yaml))

output_problems = []
skip_problems = []
for i, o in enumerate(test_data, 1):
    if o in train_data:
        skip_problems.append((i, o))
    else:
        output_problems.append(o)
print(f"Skip {len(skip_problems)} duplicate problems.")
for i, o in skip_problems:
    print(f"#{i}: {o}")
    os.system(f"sed -i '{i}d' {output_yaml}")
