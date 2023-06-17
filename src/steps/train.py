import sys
import yaml
import pandas as pd
import sys
sys.path.append('.\\src\\models\\')
from baseline import Baseline

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py train_file test_file\n'
    )
    sys.exit(1)

train_file = sys.argv[1]
test_file = sys.argv[2]

params = yaml.safe_load(open('params.yaml'))['train']
model = params['model']
method = params['method']

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

train_model = Baseline(method)
train_model.train(train_df)

train_model.dump(f'{model}_{method}')