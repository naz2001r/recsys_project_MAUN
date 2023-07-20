import numpy as np
from numpy.linalg import svd, norm
import pandas as pd
from tqdm import tqdm
from base_model import BaseModel
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent

import torch
from sentence_transformers import SentenceTransformer, util

class Hybrid_MF_STransformer(BaseModel):
    """Hybrid of Matrix Factorization ans Sentence Transformer Algorithm"""
    MODEL_NAME = 'Hybrid_MF_STransformer'

    def __init__(self, 
                 transformer_model: str = 'all-MiniLM-L6-v2',
                 bookid: str = "ISBN", 
                 userid: str = "User-ID", 
                 bookrank: str = "Book-Rating",
                 titleid: str = "Book-Title",
                 filter_treshold: int = 10) -> None:
        """
        Initialize the Hybrid_MF_STransformer model.

        Args:
            transformer_model (str, optional): The name of the transformer model to use. Defaults to 'all-MiniLM-L6-v2'.
            bookid (str, optional):   The name of the column representing the book ID. Defaults to "ISBN".
            userid (str, optional):   The name of the column representing the user ID. Defaults to "User-ID".
            bookrank (str, optional): The name of the column representing the book rating. Defaults to "Book-Rating".
            titleid (str, optional): The name of the column representing the book title. Defaults to "Book-Title".
            filter_treshold (int, optional): The minimum number of ratings a book must have to be considered. Defaults to 10.
        """
        super().__init__(self.MODEL_NAME)

        self.bookid = bookid
        self.userid = userid
        self.bookrank = bookrank
        self.titleid = titleid
        self.filter_treshold = filter_treshold
        self.model = SentenceTransformer(transformer_model)
        self.rank = None
        self.popular_books = None
        self.user_book_matrix = None
        self.vh = None
        

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the CollabFilter model using the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the book ratings data.
        """

        # Delete books with less than filter_treshold ratings
        print(f"Training {self.MODEL_NAME} model...")
        self.popular_books = df[df['Total_No_Of_Users_Rated'] > self.filter_treshold].reset_index(drop = True)

        self.rank = df.groupby([self.bookid]).agg({
            self.bookrank: "mean"
        }).reset_index()

        print("Creating user-book matrix...")
        self.user_book_matrix = self.popular_books.pivot_table(index=self.userid, columns=self.bookid, values=self.bookrank, aggfunc = len).fillna(0)
        # SVD
        print("Calculating SVD...")
        matrix = self.user_book_matrix.values
        _, _, self.vh = svd(matrix, full_matrices=False)

        self.df = df
        self.books_df = df[[self.bookid ,self.titleid]].drop_duplicates().reset_index(drop=True)

        print("Computing embeddings for books...")
        self.embeddings_db = self.model.encode(self.books_df[self.titleid].tolist())
        print("Training complete!")

    
    # Function to calculate cosine similarity
    @staticmethod
    def cosine_similarity(A, B):
        return np.dot(A,B)/(norm(A)*norm(B))
    
    # When user type the user id, with this function we will get the id of this user favourite book
    def _index_of_fav_book(self, user_id: int) -> int:
        """"
        Get the index of the user's favourite book.

        Args:
            user_id (int): The user ID.
        
        Returns:
            int: The index of the user's favourite book.
        """

        name_book = self.popular_books[self.popular_books[self.userid] == user_id]

        # If the user has not rated any book, return -1
        if name_book.empty:
            return -1
        name_book = name_book[name_book[self.bookrank] == name_book[self.bookrank].max()]
        name_book = name_book[name_book['Total_No_Of_Users_Rated'] == name_book['Total_No_Of_Users_Rated'].max()]
        name_book = name_book.iloc[0][5]
        return np.where(pd.DataFrame(self.user_book_matrix.columns)[self.bookid] == name_book)[0][0]
    
    def _get_user_predictions(self, user_id: int, top_liked_n: int = 3) -> list:
        """
        Generate Sentence Transformer similarity for the given user.

        Args:
            user_id (int): The user to generate similarity for.
            top_liked_n (int, optional): The number of top liked books to use for the prediction. Defaults to 3.

        Returns:
            pd.DataFrame: A DataFrame containing the similarity for the given user.
        """

        user_df = self.df[self.df[self.userid] == user_id].sort_values(self.bookrank, ascending=False)
        user_books = user_df[self.bookid][:top_liked_n]

        user_predictions = []
        for book_id in user_books:
            book_idx = self.books_df[self.books_df[self.bookid] == book_id].index[0]
            book_embedding = self.embeddings_db[book_idx]
            cosine_scores = util.pytorch_cos_sim(book_embedding, self.embeddings_db)[0]
            top_results = torch.topk(cosine_scores, k=int(0.7*cosine_scores.shape[0]))
            for score, idx in zip(top_results[0], top_results[1]):
                book = self.books_df.iloc[idx.item()][self.bookid]
                user_predictions.append({self.bookid: book, 'predictions': score.item()})

        pred_df = pd.DataFrame(user_predictions).sort_values('predictions', ascending=False)
        pred_df = pred_df.groupby([self.bookid]).agg({
            'predictions': 'mean'
        }).reset_index().sort_values('predictions', ascending=False)
        return pred_df

    # Recommendations
    def _recommend(self, user_book: int, user: int, k: int, top_liked_n: int = 3) -> list:
        """"
        Generate recommendations for the given user.
        
        Args:
            user_book (int): The index of the user's favourite book.
            user (int): The user ID.
            k (int): The number of top recommendations to return.
            top_liked_n (int, optional): The number of top liked books to use for the prediction. Defaults to 3.
        
        Returns:
            list: A list of recommended book ISBN for the given user.
        """

        # Caclucate cosine similarity between vectors with books
        sim = []
        for col in range(self.vh.shape[1]):
            similarity = self.cosine_similarity(self.vh[:,user_book], self.vh[:,col])
            sim.append(similarity)
        # Find most relevant books 
        sim = pd.DataFrame(sim)
        sim[self.bookid] = self.user_book_matrix.columns

        stranformer_pred = self._get_user_predictions(user, top_liked_n)
        
        # Join by bookid
        sim = sim.merge(stranformer_pred, on=self.bookid)
        sim.fillna(sim.predictions.min(), inplace=True)

        # similarities sum
        sim[0] = sim[0] * sim['predictions']

        recommend = sim.sort_values(by=0, ascending=False).reset_index().loc[1:]
        indexes = list(recommend.set_index('index').index)

        # Find books that user has already read
        user_books = self.popular_books[self.popular_books[self.userid] == user][self.bookid].tolist()
        recom_book = []
        for i in indexes:
            book = pd.DataFrame(self.user_book_matrix.columns).iloc[i,0]
            if book not in user_books:
                recom_book.append(book)
            if len(recom_book) == k:
                break
        
        return recom_book
    
    def _process_user(self, user: int, k: int, top_liked_n: int = 3) -> list:
        """"
        Process a single user and generate recommendations.

        Args:
            user (int): The user ID.
            k (int): The number of top recommendations to return.
            top_liked_n (int, optional): The number of top liked books to use for the prediction. Defaults to 3.
        
        Returns:
            list: A list of recommended book ISBN for the given user.
        """

        user_book = self._index_of_fav_book(user)
        if user_book == -1:
            #print(f"User {user} has not rated any book in the dataset. Returning top {k} popular books.")
            return self.rank.nlargest(k, self.bookrank)[self.bookid].tolist()
        return self._recommend(user_book, user, k, top_liked_n)
    
    def predict(self, users: np.array, k: int = 3, top_liked_n: int = 3) -> np.array:
        """
        Generate predictions for the given users.

        Args:
            users (np.array):  An array of user IDs.
            k (int, optional): The number of top recommendations to return. Defaults to 3.
            top_liked_n (int, optional): The number of top liked books to use for the prediction. Defaults to 3.

        Returns:
            np.array: An array of predicted book IDs for the given users.
        """
            
        predictions = []
        with ThreadPoolExecutor() as executor:
            futures = []
            print("Loading executor")
            for user in tqdm(users):
                future = executor.submit(self._process_user, user, k, top_liked_n)
                futures.append(future)
            
            print("Collecting results from executor")
            with tqdm(total=len(users)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    predictions.append(result)
                    pbar.update(1)
        return np.array(predictions)

    
if __name__ == "__main__":
    # Run this code to test if script is not failing
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/prepared/preprocessed.csv")#[:10000]
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    model = Hybrid_MF_STransformer()
    model.train(df_train)
    print(model.predict(df_test[model.userid].unique(), k=5))