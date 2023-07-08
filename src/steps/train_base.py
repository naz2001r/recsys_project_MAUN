import dvc.api
import sys
import pandas as pd
sys.path.append('./src/models/')


class TrainStepABC():
    def __init__(self, model_name):
        params = dvc.api.params_show()

        self._model = model_name
        self._method = params['train']['method']
        self._filter_treshold = params['filter_treshold']
        

    def __read_files(self) -> tuple:
        train_file = sys.argv[1]
        test_file = sys.argv[2]

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        return (train_df, test_df)


    def train(self):
        print("Preparing training. Fasten your seatbelt!")
        train_df, test_file = self.__read_files()

        if self._model == 'baseline':
            from baseline import Baseline
            train_model = Baseline(self._method)

        elif self._model == 'CollaborativeFiltering':
            from collab_filter_model import CollaborativeFiltering
            train_model = CollaborativeFiltering(filter_treshold=self._filter_treshold)

        elif self._model == 'MatrixFactorization':
            from matrix_factorization import MatrixFactorization
            train_model = MatrixFactorization(filter_treshold=self._filter_treshold)

        print("Started training. Prepare to takeoff!")
        train_model.train(train_df)
        print("Training finished. Applause for the captain!")
        train_model.dump(f'{self._model}_{self._method}' if self._method else self._model)
        print("Model dumped. Thanks for choosing us!")
        