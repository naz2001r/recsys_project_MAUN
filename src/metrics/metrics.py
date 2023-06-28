import numpy as np
from itertools import product


class ReccomenderMetrics:
    """All metrics are taken from"""
        
    def apk(self, true: np.array, pred: np.array, k_max:int=0) -> float:
        """
        Average Precision at k (AP@k)

        Args:
            true (np.array): The true values.
            pred (np.array): The predicted values.
            k_max (int, optional): The maximum number of predictions to consider. Defaults to 0.

        Returns:
            float: The average precision at k.
        """
        if k_max != 0:
            pred = pred[:k_max]
            true = true[:k_max]


        relevant_predictions = 0
        sum = 0

        for i, pred_item in enumerate(pred):
            k = i+1 
    
            if pred_item in true:
                relevant_predictions += 1
                sum += relevant_predictions/k

        return sum/len(true)
    
    def mapk(self, actual: np.array, predictions: np.array, k:int=0) -> float:
        """"
        Mean Average Precision at k (MAP@k)
        
        Args:
            actual (np.array): The true values.
            predictions (np.array): The predicted values.
            k (int, optional): The maximum number of predictions to consider. Defaults to 0.

        Returns:
            float: The mean average precision at k.
        """
        return np.mean([np.mean([self.apk(a,p,k) for a,p in product(act, pred)]) for act, pred in zip(actual, predictions)])