import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
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
user_column = params['user_column']

os.makedirs(output_dir, exist_ok=True)

input_folder = sys.argv[1]
dataset = pd.read_csv(os.path.join(input_folder, 'preprocessed.csv'))

dataset = dataset.groupby(book_column).filter(lambda x: len(x) >= filter_treshold)

train, test = train_test_split(
    dataset.index,
    train_size=1-test_size,
    test_size=test_size,
    random_state=seed,
    stratify=dataset[stratify_column]
)

val_ratio = round((1 - train_size - test_size) / train_size, 2)
train, val = train_test_split(
    dataset.loc[train].index,
    train_size=1-val_ratio,
    test_size=val_ratio,
    random_state=seed,
    stratify=dataset.loc[train][stratify_column]
)

test = dataset.loc[test]

missing_users = [user_id for user_id in dataset.loc[val][user_column].unique() if user_id not in dataset.loc[train][user_column].unique()]
val_temp = dataset.loc[val]
missing_users_index = val_temp.loc[val_temp[user_column].isin(missing_users)].index

train = dataset.loc[np.concatenate((train, missing_users_index))]
val = val_temp.loc[~val_temp[user_column].isin(missing_users)]

train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
