import numpy as np


class ReccomenderMetrics:
    """All metrics are taken from"""
        
    def apk(self, actual, predicted, k=10):
        """
        Computes the average precision at k.
        This function computes the average precision at k between two lists of
        items.
        
        Args:
            actual(list): A list of elements that are to be predicted
            predicted(list): A list of predicted elements
            k (int, optional): The maximum number of predicted elements. Default is 10.
        
        Returns:
            float: The average precision at k over the input lists
        """
        
        score = 0.0
        num_hits = 0.0
        for i,p in enumerate(predicted[:k]):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)
        
        if not actual:
            return 0.0
        
        return score / min(len(actual), k)

    def mapk(self, actual, predicted, k=10):
        """
        Computes the mean average precision at k.
        This function computes the mean average precision at k between two lists
        of lists of items.
        
        Args:
            actual(list): A list of lists of elements that are to be predicted
            predicted(list): A list of lists of predicted elements
            k (int, optional): The maximum number of predicted elements. Default is 10.
        
        Returns:
            float: The mean average precision at k over the input lists
        """
        
        return np.mean([self.apk(a,p,k) for a,p in zip(actual, predicted)])
    
    def precisionk(self, actual, predicted, k=10):
        """
        Computes the precision at k.
        This function computes the precision at k between two lists
        of items.
        
        Args:
            actual(list): A list of elements that are to be predicted
            predicted(list): A list of predicted elements
            k (int, optional): The maximum number of predicted elements. Default is 10.
        
        Returns:
            float: The precision at k over the input lists
        """
        
        return np.mean([1 if p in actual else 0 for p in predicted[:k]])
    
    def recallk(self, actual, predicted, k=10):
        """
        Computes the recall at k.
        This function computes the recall at k between two lists
        of items.
        
        Args:
            actual(list): A list of elements that are to be predicted
            predicted(list): A list of predicted elements
            k (int, optional): The maximum number of predicted elements. Default is 10.
        
        Returns:
            float: The recall at k over the input lists
        """
        
        return np.sum([1 if p in actual else 0 for p in predicted[:k]]) / len(actual)
    
    def _dcgk(self, actual, predicted, k=10):
        """
        Computes the discounted cumulative gain at k.
        This function computes the dcg at k between two lists
        of items.
        
        Args:
            actual(list): A list of elements that are to be predicted
            predicted(list): A list of predicted elements
            k (int, optional): The maximum number of predicted elements. Default is 10.
        
        Returns:
            float: The ndcg at k over the input lists
        """
        
        score = 0.0
        for i,p in enumerate(predicted[:k]):
            if p in actual:
                score += 1 / np.log2(i+2)
        
        return score
    
    def ndcgk(self, actual, predicted, k=10):
        """
        Computes the normalized discounted cumulative gain at k.
        This function computes the ndcg at k between two lists
        of items.
        
        Args:
            actual(list): A list of elements that are to be predicted
            predicted(list): A list of predicted elements
            k (int, optional): The maximum number of predicted elements. Default is 10.
        
        Returns:
            float: The ndcg at k over the input lists
        """
            
        dcg_max = self._dcgk(actual, actual, k)
        dcg = self._dcgk(actual, predicted, k)
        
        return dcg / dcg_max
    
    def coverage(self, predicted, catalog):
        """
        Computes the coverage.
        This function computes the coverage between two lists
        of items.
        
        Args:
            predicted(list): A list of predicted elements
            catalog(list): A list of all elements
        
        Returns:
            float: The coverage over the input lists
        """
        
        return len(set(np.concatenate(predicted))) / len(set(catalog))