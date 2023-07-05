import yaml
import pandas as pd
import sys
sys.path.append('./src/models/')

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
filter_treshold = params['filter_treshold']

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

if model == 'baseline':
    from baseline import Baseline
    train_model = Baseline(method)

elif model == 'CollaborativeFiltering':
    from collab_filter_model import CollaborativeFiltering
    train_model = CollaborativeFiltering(filter_treshold=filter_treshold)

elif model == 'MatrixFactorization':
    from matrix_factorization import MatrixFactorization
    train_model = MatrixFactorization(filter_treshold=filter_treshold)

else:
    pass

train_model.train(train_df)
print(1)

train_model.dump(f'{model}_{method}' if method else model)