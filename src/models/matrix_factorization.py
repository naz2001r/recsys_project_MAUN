import numpy as np
from numpy.linalg import svd, norm
import pandas as pd
from tqdm import tqdm
from base_model import BaseModel
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent

class MatrixFactorization(BaseModel):
    """Matrix Factorization Algorithm"""
    MODEL_NAME = 'MatrixFactorization'

    def __init__(self, 
                 bookid: str = "ISBN", 
                 userid: str = "User-ID", 
                 bookrank: str = "Book-Rating",
                 filter_treshold: int = 10) -> None:
        """
        Initialize the MatrixFactorization model.

        Args:
            bookid (str, optional):   The name of the column representing the book ID. Defaults to "ISBN".
            userid (str, optional):   The name of the column representing the user ID. Defaults to "User-ID".
            bookrank (str, optional): The name of the column representing the book rating. Defaults to "Book-Rating".
            filter_treshold (int, optional): The minimum number of ratings a book must have to be considered. Defaults to 10.
        """
        super().__init__(self.MODEL_NAME)

        self.bookid = bookid
        self.userid = userid
        self.bookrank = bookrank
        self.filter_treshold = filter_treshold
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
        print("Training model...")
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
        name_book=name_book.iloc[0][3]
        
        return np.where(pd.DataFrame(self.user_book_matrix.columns)[self.bookid] == name_book)[0][0]

    # Recommendations
    def _recommend(self, user_book: int, k: int) -> list:
        """"
        Generate recommendations for the given user.
        
        Args:
            user_book (int): The index of the user's favourite book.
            k (int): The number of top recommendations to return.
        
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
        recommend = sim.sort_values(by=0, ascending=False).reset_index().loc[1:k]
        indexes = list(recommend.set_index('index').index)
        recom_book = []
        for i in indexes:
            recom_book.append(pd.DataFrame(self.user_book_matrix.columns).iloc[i,0])
        
        return recom_book
    
    def _process_user(self, user: int, k: int) -> list:
        """"
        Process a single user and generate recommendations.

        Args:
            user (int): The user ID.
            k (int): The number of top recommendations to return.
        
        Returns:
            list: A list of recommended book ISBN for the given user.
        """

        user_book = self._index_of_fav_book(user)
        if user_book == -1:
            #print(f"User {user} has not rated any book in the dataset. Returning top {k} popular books.")
            return self.rank.nlargest(k, self.bookrank)[self.bookid].tolist()
        return self._recommend(user_book, k)
    
    def predict(self, users: np.array, k: int = 3) -> np.array:
        """
        Generate predictions for the given users.

        Args:
            users (np.array):  An array of user IDs.
            k (int, optional): The number of top recommendations to return. Defaults to 3.

        Returns:
            np.array: An array of predicted book IDs for the given users.
        """
            
        predictions = []
        with ThreadPoolExecutor() as executor:
            futures = []
            print("Loading executor")
            for user in tqdm(users):
                future = executor.submit(self._process_user, user, k)
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

    model = MatrixFactorization()
    model.train(df_train)
    print(model.predict(df_test[model.userid].unique(), k=5))