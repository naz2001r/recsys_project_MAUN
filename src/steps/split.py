import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import dvc.api


if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 split.py input-dir output-dir\n'
    )
    sys.exit(1)

output_dir = sys.argv[2]

params = dvc.api.params_show()

train_size = params['split']['train_size']
test_size = params['split']['test_size']
seed = params['split']['seed']
stratify_column = params['split']['stratify_column']
filter_treshold = params['filter_treshold']
book_column = params['book_column']

os.makedirs(output_dir, exist_ok=True)

input_folder = sys.argv[1]
dataset = pd.read_csv(os.path.join(input_folder, 'preprocessed.csv'))

dataset = dataset.groupby(book_column).filter(lambda x: len(x) >= filter_treshold)

train, test = train_test_split(
    dataset.index,
    train_size=train_size,
    test_size=test_size,
    random_state=seed,
    stratify=dataset[stratify_column]
)

train = dataset.loc[train]
test = dataset.loc[test]
val = dataset[(~dataset.index.isin(train.index)) & (~dataset.index.isin(test.index))]


train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
