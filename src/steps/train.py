import yaml
import pandas as pd
import sys
sys.path.append('./src/models/')
import dvc.api

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py train_file test_file\n'
    )
    sys.exit(1)

train_file = sys.argv[1]
test_file = sys.argv[2]

params = dvc.api.params_show()

model = params['train']['model']
method = params['train']['method']
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

elif model == 'STransformer':
    from stransformer_content_based import STransformContentBase
    train_model = STransformContentBase(transformer_model = params['train']['transformer_model'])

elif model == 'HybridNN_Recommender':
    from hybrid_nn_model import HybridNN_Recommender
    train_model = HybridNN_Recommender(transformer_model = params['train']['transformer_model'], 
                                       num_epochs = params['train']['num_epochs'])

else:
    raise Exception(f'Model {model} not found.')

train_model.train(train_df)

train_model.dump(f'{model}_{method}' if method else model)