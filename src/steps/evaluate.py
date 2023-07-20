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

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 evaluate.py eval_file full_file model_name\n'
    )
    sys.exit(1)

params = dvc.api.params_show()
k_list = params['evaluate']['k_list']
user_column = params['user_column']
book_column = params['book_column']

eval_file = sys.argv[1]
eval_df = pd.read_csv(eval_file)

file_full = sys.argv[2]
df_full = pd.read_csv(file_full)

model_name = sys.argv[3]

model_dump = ''

if len(sys.argv) == 5:
    model_dump = sys.argv[4]
else:
    list_of_files = glob.glob(f'./data/models/{model_name}/*.pkl', recursive=True)
    print('Next models will be evaluated:')
    print([f.split("/")[-1].split("_")[0] for f in list_of_files])

model_dump = list_of_files[0]

metrics_obj = ReccomenderMetrics()
metrics = {}

print(f'Model evaluation started {model_dump}')
eval_model = BaseModel('base').load(model_dump)
metrics[f'{eval_model.name()}'] = {}
    
users = eval_df[user_column].unique()
pred = eval_model.predict(users, k=max(k_list))
true = [eval_df[eval_df[user_column] == user_id].sort_values(by="Book-Rating", ascending=False)[book_column].tolist() for user_id in users]

for k in k_list:
    metrics[f'{eval_model.name()}'][f'map@{k}'] = metrics_obj.mapk(true, pred, k)
    metrics[f'{eval_model.name()}'][f'precision@{k}'] = np.mean([metrics_obj.precisionk(true[i][:k], pred[i][:k], k) for i in range(len(users))])
    metrics[f'{eval_model.name()}'][f'recall@{k}'] = np.mean([metrics_obj.recallk(true[i][:k], pred[i][:k], k) for i in range(len(users))])
    metrics[f'{eval_model.name()}'][f'ndcg@{k}'] = np.mean([metrics_obj.ndcgk(true[i][:k], pred[i][:k], k) for i in range(len(users))])
    
metrics[f'{eval_model.name()}'][f'coverage'] = metrics_obj.coverage(true, df_full["ISBN"].unique())    

os.makedirs('data/eval', exist_ok=True)
metrics_file = f'metrics_{model_name}.json'
with open(os.path.join('data/eval', metrics_file), "w") as mf:
    json.dump(metrics, mf)