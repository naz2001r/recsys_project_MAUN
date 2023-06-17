import numpy as np
from itertools import product


class ReccomenderMetrics:
        
    def apk(self, true: np.array, pred: np.array, k_max:int=0) -> float:
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
        return np.mean([np.mean([self.apk(a,p,k) for a,p in product(act, pred)]) for act, pred in zip(actual, predictions)])