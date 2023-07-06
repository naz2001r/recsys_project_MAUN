import yaml
import os
import glob
import pandas as pd
import numpy as np
import json

import sys
sys.path.append('./src/models/')
sys.path.append('./src/metrics/')

from base_model import BaseModel
from metrics import ReccomenderMetrics
import dvc.api

if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 evaluate.py eval_file \n'
    )
    sys.exit(1)

params = dvc.api.params_show()
k_list = params['evaluate']['k_list']
user_column = params['user_column']
book_column = params['book_column']

eval_file = sys.argv[1]
eval_df = pd.read_csv(eval_file)

model_dump = ''

if len(sys.argv) == 3:
    model_dump = sys.argv[2]
else:
    list_of_files = glob.glob('./data/models/**/*.pkl', recursive=True)
    model_dump = max(list_of_files, key=os.path.getctime)
    print(model_dump)
eval_model = BaseModel('base').load(model_dump)

metrics_obj = ReccomenderMetrics()
metrics = {}
for k in k_list:
    users = eval_df[user_column].unique()
    pred = eval_model.predict(users, k=k)
    true = [eval_df[eval_df[user_column] == user_id].sort_values(by="Book-Rating", ascending=False)[book_column].tolist() for user_id in users]
    metrics[f'map@{k}'] = metrics_obj.mapk(true, pred, k)
    metrics[f'precision@{k}'] = np.mean([metrics_obj.precisionk(true[i], pred[i], k) for i in range(len(users))])
    metrics[f'recall@{k}'] = np.mean([metrics_obj.recallk(true[i], pred[i], k) for i in range(len(users))])
    metrics[f'ndcg@{k}'] = np.mean([metrics_obj.ndcgk(true[i], pred[i], k) for i in range(len(users))])
metrics[f'coverage'] = metrics_obj.coveragek(true, pred, k)

os.makedirs('data/eval', exist_ok=True)
with open(os.path.join('data/eval', "metrics.json"), "w") as mf:
    json.dump(metrics, mf)