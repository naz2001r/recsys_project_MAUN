import sys
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 split.py input-dir output-dir\n'
    )
    sys.exit(1)

output_dir = sys.argv[2]

params = yaml.safe_load(open('params.yaml'))['split']
train_size = params['train_size']
test_size = params['test_size']
seed = params['seed']
stratify_column = params['stratify_column']

os.makedirs(output_dir, exist_ok=True)

input_folder = sys.argv[1]
dataset = pd.read_csv(os.path.join(input_folder, 'preprocessed.csv'))

train, test = train_test_split(
    dataset.index,
    train_size=train_size,
    test_size=test_size,
    random_state=seed
)

train = dataset.loc[train]
test = dataset.loc[test]
val = dataset[(~dataset.index.isin(train.index)) & (~dataset.index.isin(test.index))]


train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
