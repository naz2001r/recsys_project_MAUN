import os
import pickle
import time  


class BaseModel:
    BASE_PATH: str = 'data\\models'

    def __init__(self, method_name: str):
        self._method_name = method_name

    def predict(self) -> None:
        raise NotImplementedError()
    
    def train(self) -> None:
        raise NotImplementedError()
    
    def load(self, file_name: str) -> object:
        return pickle.load(open(file_name, 'rb'))
    
    def dump(self, file_name: str) -> None:
        os.makedirs(os.path.join(self.BASE_PATH, self._method_name), exist_ok=True)
        time_stamp = time.time_ns()
        pickle.dump(self, open(os.path.join(self.BASE_PATH, self._method_name, file_name + f'_{time_stamp}.pkl'), 'wb'))
