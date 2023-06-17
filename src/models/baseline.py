import numpy as np
import pandas as pd
from base_model import BaseModel

class Baseline(BaseModel):
    MODEL_NAME = 'baseline'
    METHODS = ['mean', 'sum']

    def __init__(self, method: str, bookid: str = "ISBN", bookrank: str = "Book-Rating") -> None:
        """
        Initialize the Baseline model.

        Args:
            method (str):             The method for aggregation. Must be one of ['avg', 'sum'].
            bookid (str, optional):   The name of the column representing the book ID. Defaults to "ISBN".
            bookrank (str, optional): The name of the column representing the book rating. Defaults to "Book-Rating".
        
        Raises:
            ValueError: If the method is not one of the available methods.
        """

        super().__init__(f'{self.MODEL_NAME}_{method}')

        if method in self.METHODS:
            self.method = method
        else:
            raise ValueError(f"Unknown method '{method}'. Must be one of {self.METHODS}.")
        
        self.rank = None
        self.bookid = bookid
        self.bookrank = bookrank

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the Baseline model using the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the book ratings data.
        """
        self.rank = df.groupby([self.bookid]).agg({
            self.bookrank: self.method
        }).reset_index()

    def predict(self, users: np.array, k: int = 3) -> np.array:
        """
        Generate predictions for the given users.

        Args:
            users (np.array):  An array of user IDs.
            k (int, optional): The number of top recommendations to return. Defaults to 3.

        Returns:
            np.array: An array of predicted book IDs for the given users.
        """
        return np.array([self.rank.nlargest(k, self.bookrank)[self.bookid].to_list()] * len(users))

