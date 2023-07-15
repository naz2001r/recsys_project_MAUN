import dvc.api
import sys
import pandas as pd
import os
sys.path.append('./src/models/')


class TrainStepABC():
    def __init__(self, model_name):
        params = dvc.api.params_show()

        self._model = model_name
        self._method = params['train']['method']
        self._filter_treshold = params['filter_treshold']
        self._transformer_model = params['train']['transformer_model']
        self._num_epochs = params['train']['num_epochs']
        self._patience = params['train']['patience']
        self._min_delta = params['train']['min_delta']
        self._train_file = params['train']['train_file']
        self._val_file = params['train']['val_file']
        

    def __read_files(self) -> tuple:
        split_dir = sys.argv[1]

        train_df = pd.read_csv(os.path.join(split_dir, self._train_file))
        val_df = pd.read_csv(os.path.join(split_dir, self._val_file))
        return (train_df, val_df)


    def train(self):
        print("Preparing training. Fasten your seatbelt!")
        train_df, val_df = self.__read_files()

        if self._model == 'baseline':
            from baseline import Baseline
            train_model = Baseline(self._method)

        elif self._model == 'CollaborativeFiltering':
            from collab_filter_model import CollaborativeFiltering
            train_model = CollaborativeFiltering(filter_treshold=self._filter_treshold)

        elif self._model == 'MatrixFactorization':
            from matrix_factorization import MatrixFactorization
            train_model = MatrixFactorization(filter_treshold=self._filter_treshold)

        elif self._model == 'STransformer':
            from stransformer_content_based import STransformContentBase
            train_model = STransformContentBase(transformer_model = self._transformer_model)

        elif self._model == 'HybridNN_Recommender':
            from hybrid_nn_model import HybridNN_Recommender
            train_model = HybridNN_Recommender(transformer_model = self._transformer_model, 
                                               num_epochs = self._num_epochs,
                                               patience = self._patience,
                                               min_delta = self._min_delta)
            
        elif self._model == 'ContentBasedFiltering':
            from content_based_filtering import ContentBasedFiltering
            train_model = ContentBasedFiltering(filter_treshold=self._filter_treshold)

        print("Started training. Prepare to takeoff!")
        if self._model == 'HybridNN_Recommender':
            train_model.train(train_df, val_df)
        else:
            train_model.train(train_df)
        print("Training finished. Applause for the captain!")
        train_model.dump(f'{self._model}_{self._method}' if self._method else self._model)
        print("Model dumped. Thanks for choosing us!")
        