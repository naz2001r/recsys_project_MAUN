import numpy as np
from numpy.linalg import svd, norm
import pandas as pd
from tqdm import tqdm
from base_model import BaseModel
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent

import torch
from sentence_transformers import SentenceTransformer, util


class STransformContentBase(BaseModel):
    """Sentence Transformer Content Based Algorithm"""

    MODEL_NAME = 'STransformContentBase'

    def __init__(self,
                 transformer_model: str = 'all-MiniLM-L6-v2',
                 bookid: str = "ISBN",
                 userid: str = "User-ID",
                 bookrank: str = "Book-Rating",
                 titleid: str = "Book-Title") -> None:
        """
        Initialize the STransformContentBase model.

        Args:
            transformer_model (str, optional): The name of the transformer model to use. Defaults to 'all-MiniLM-L6-v2'.
            bookid (str, optional):   The name of the column representing the book ID. Defaults to "ISBN".
            userid (str, optional):   The name of the column representing the user ID. Defaults to "User-ID".
            bookrank (str, optional): The name of the column representing the book rating. Defaults to "Book-Rating".
            titleid (str, optional): The name of the column representing the book title. Defaults to "Book-Title".
        """
        super().__init__(self.MODEL_NAME)

        self.bookid = bookid
        self.userid = userid
        self.bookrank = bookrank
        self.titleid = titleid

        self.model = SentenceTransformer(transformer_model)
        self.df = None
        self.books_df = None
        self.embeddings_db = None
        self.rank = None

    
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the model.

        Args:
            df (pd.DataFrame): The dataframe to train the model on.
        """
        print(f"Training {self.MODEL_NAME} model...")
        print("Computing book rank...")
        self.rank = df.groupby([self.bookid]).agg({
            self.bookrank: "mean"
        }).reset_index()

        self.df = df
        self.books_df = df[[self.bookid ,self.titleid]].drop_duplicates().reset_index(drop=True)

        print("Computing embeddings for books...")
        self.embeddings_db = self.model.encode(self.books_df[self.titleid].tolist())
        print("Done!")

    def _get_user_predictions(self, user_id: int, top_n: int, top_liked_n: int = 3) -> list:
        """
        Generate predictions for a single user.

        Args:
            user_id (int): The user to generate predictions for.
            top_n (int): The number of predictions to generate.
            top_liked_n (int, optional): The number of top liked books to use for the prediction. Defaults to 3.

        Returns:
            list: A list of predictions for the given user.
        """
        user_df = self.df[self.df[self.userid] == user_id].sort_values(self.bookrank, ascending=False)
        user_books = user_df[self.bookid][:top_liked_n]

        if len(user_books) == 0:
            #print(f"User {user} has not rated any book in the dataset. Returning top {top_n} popular books.")
            return self.rank.nlargest(top_n, self.bookrank)[self.bookid].tolist()

        user_predictions = []
        for book_id in user_books:
            book_idx = self.books_df[self.books_df[self.bookid] == book_id].index[0]
            book_embedding = self.embeddings_db[book_idx]
            cosine_scores = util.pytorch_cos_sim(book_embedding, self.embeddings_db)[0]
            top_results = torch.topk(cosine_scores, k=int(0.8*cosine_scores.shape[0]))
            for score, idx in zip(top_results[0], top_results[1]):
                book = self.books_df.iloc[idx.item()][self.bookid]
                user_predictions.append({self.bookid: book, 'predictions': score.item()})

        pred_df = pd.DataFrame(user_predictions).sort_values('predictions', ascending=False)
        pred_df = pred_df.groupby([self.bookid]).agg({
            'predictions': 'mean'
        }).reset_index().sort_values('predictions', ascending=False)

        return [pred for pred in pred_df[self.bookid].values.tolist() 
                    if pred not in user_df[self.bookid].values.tolist()][:top_n]

    def predict(self, users: np.array, k: int = 3, top_liked_n: int = 3) -> np.array:
        """
        Generate predictions for the given users.

        Args:
            users (np.array):  An array of user IDs.
            k (int, optional): The number of top recommendations to return. Defaults to 3.
            top_liked_n (int, optional): The number of top liked books to use for the prediction for each user. Defaults to 3.

        Returns:
            np.array: An array of predicted book IDs for the given users.
        """
        predictions = []
        with ThreadPoolExecutor() as executor:
            futures = []
            print("Loading executor")
            for user in tqdm(users):
                future = executor.submit(self._get_user_predictions, user, k, top_liked_n)
                futures.append(future)
            
            print("Collecting results from executor")
            with tqdm(total=len(users)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    predictions.append(result)
                    pbar.update(1)
        return np.array(predictions)
