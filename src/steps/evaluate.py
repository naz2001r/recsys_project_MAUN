import yaml
import os
import glob
import pandas as pd
import numpy as np
import json

import sys
sys.path.append('.\\src\\models\\')
sys.path.append('.\\src\\metrics\\')
from base_model import BaseModel
from metrics import ReccomenderMetrics

if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 evaluate.py eval_file \n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))['evaluate']
k_list = params['k_list']
user_column = params['user_column']
book_column = params['book_column']

eval_file = sys.argv[1]
eval_df = pd.read_csv(eval_file)

model_dump = ''

if len(sys.argv) == 3:
    model_dump = sys.argv[2]
else:
    list_of_files = glob.glob('.\\data\\models\\**\\*.pkl', recursive=True)
    model_dump = max(list_of_files, key=os.path.getctime)

eval_model = BaseModel('base').load(model_dump)
pred = eval_model.predict(eval_df[user_column].unique(), k=max(k_list))

metrics_obj = ReccomenderMetrics()
metrics = {}
for k in k_list:
    metrics[f'map@{k}'] = metrics_obj.mapk([eval_df[eval_df[user_column] == user_id][book_column].unique().tolist() for user_id in eval_df[user_column].unique()], pred, k)

os.makedirs('data\\eval', exist_ok=True)
with open(os.path.join('data\\eval', "metrics.json"), "w") as mf:
    json.dump(metrics, mf)