import os
import pickle
import time  


class BaseModel:
    """Base class for all models."""
    MODEL_NAME: str = 'base'
    BASE_PATH: str = 'data/models'

    def __init__(self, method_name: str):
        """"
        Initialize the BaseModel.
        
        Args:
            method_name (str): The name of the method used for the model.
        """
        self._method_name = method_name

    
    def name(self):
        return self.MODEL_NAME

    def predict(self) -> None:
        """Generate predictions for the given users."""
        raise NotImplementedError()
    
    def train(self) -> None:
        """Train the model."""
        raise NotImplementedError()
    
    def load(self, file_name: str) -> object:
        """
        Load a model from a file.
        
        Args:
            file_name (str): The name of the file to load the model from.
        
        Returns:
            object: The loaded model.
        """
        return pickle.load(open(file_name, 'rb'))
    
    def dump(self, file_name: str) -> None:
        """
        Dump the model to a file.

        Args:
            file_name (str): The name of the file to dump the model to.
        """
        os.makedirs(os.path.join(self.BASE_PATH, self._method_name), exist_ok=True)
        time_stamp = time.time_ns()
        pickle.dump(self, open(os.path.join(self.BASE_PATH, self._method_name, file_name + f'_{time_stamp}.pkl'), 'wb'))
