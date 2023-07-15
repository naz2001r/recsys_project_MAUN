import numpy as np
import pandas as pd
import concurrent

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from base_model import BaseModel
from concurrent.futures.thread import ThreadPoolExecutor

import warnings
warnings.simplefilter('ignore')

from tqdm import tqdm
tqdm.pandas()

class ContentBasedFiltering(BaseModel):
    """Content Based Filtering Algorithm"""
    MODEL_NAME = 'ContentBasedFiltering'

    def __init__(self, 
                 bookid: str = "ISBN", 
                 userid: str = "User-ID", 
                 bookrank: str = "Book-Rating",
                 booktitle: str = "Book-Title",
                 filter_treshold: int = 10) -> None:
        """
        Initialize the CollabFilter model.

        Args:
            bookid (str, optional):   The name of the column representing the book ID. Defaults to "ISBN".
            userid (str, optional):   The name of the column representing the user ID. Defaults to "User-ID".
            bookrank (str, optional): The name of the column representing the book rating. Defaults to "Book-Rating".
            booktitle (str, optional): The name of the column representing the book title. Defaults to "Book-Title".
            filter_treshold (int, optional): The minimum number of ratings a book must have to be considered. Defaults to 100.
        """
        super().__init__(self.MODEL_NAME)

        self.bookid = bookid
        self.userid = userid
        self.bookrank = bookrank
        self.booktitle = booktitle
        self.filter_treshold = filter_treshold
        self.rank = None
        self.tf_vectorizer = None
        self.tf_matrix = None

    def create_cosine_matrix(self, tf_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix between books titles.

        Parameters:
            tf_matrix (np.ndarray): Term frequency-inverse document frequency matrix representing the books.

        Returns:
            np.ndarray: Cosine similarity matrix.
        """
        cosine_matrix = cosine_similarity(tf_matrix, tf_matrix)
        return cosine_matrix
    
    def get_weighted_similar_books_df(self,
                                      user_data: pd.DataFrame,
                                      cosine_sim_matrix: np.ndarray) -> pd.DataFrame:
        """
        Get a DataFrame of weighted similar books based on user data and cosine similarity matrix.

        Parameters:
            user_data (pd.DataFrame): DataFrame containing the user's book data.
            cosine_sim_matrix (np.ndarray): Cosine similarity matrix between books.

        Returns:
            pd.DataFrame: DataFrame of weighted similar books.
        """
        similar_books = []

        # Iterate over each user book and each book in the dataset
        for i in range(len(user_data)):
            for j in range(len(self.df)):
                similarity = cosine_sim_matrix[i, j]

                # Exclude books with similarity score of 0 or 1
                if similarity != 0 or similarity != 1:
                    user_book = user_data[self.booktitle].iloc[i]
                    user_rating = user_data[self.bookrank].iloc[i]
                    other_book = self.df[self.booktitle].iloc[j]
                    bookid = self.df[self.bookid].iloc[j]

                    # Store the book, similarity score, and user rating in the list
                    similar_books.append((user_book, other_book, similarity, user_rating, bookid))

        # Create a DataFrame from the list of similar books
        similar_books_df = pd.DataFrame(similar_books, columns=['User-Book', 'Book', 'Similarity', 'User-Rating', self.bookid])

        # Compute the weighted similarity score by multiplying similarity and user rating
        similar_books_df['Weighted-Similarity'] = similar_books_df['Similarity'] * similar_books_df['User-Rating']

        return similar_books_df

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the Content Based Filtering model using the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the book ratings data.
        """

        print(f"Training {self.MODEL_NAME} model...")

        # Delete books with less than filter_treshold ratings
        print('Data filtering...')
        df = df[df['Total_No_Of_Users_Rated'] >= self.filter_treshold]
        self.df = df.drop_duplicates(subset=[self.bookid, self.booktitle]).reset_index(drop = True)
        
        print('TF_IDF vectorization...')
        self.tf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        self.tf_matrix = sparse.csr_matrix(self.tf_vectorizer.fit_transform(df[self.booktitle]))

        print("Training completed!")

    def _get_user_recommendations(self, user_id: int, top_n: int = 3) -> list:
        """
        Get the top n book recommendations for a given user.

        Parameters:
            user_id (int): ID of the user.
            top_n (int): Number of recommendations to return.

        Returns:
            list: List of book IDs representing the top recommendations for the user.
        """

        # Filter data for the specified user
        user_data = self.df[self.df[self.userid] == user_id]

        # Get the book titles rated by the user
        user_books = user_data[self.booktitle].tolist()

        if len(user_books) == 0:
            return self.df.nlargest(top_n, 'Avg_Rating')[self.bookid].tolist()
    
        # tf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        user_tf_matrix = self.tf_vectorizer.transform(user_data[self.booktitle])
        cosine_similarity_matrix = cosine_similarity(user_tf_matrix, self.tf_matrix)

        # Create a pairwise DataFrame with user books, all books, and similarity scores
        similar_books_df = self.get_weighted_similar_books_df(user_data, cosine_similarity_matrix)
        
        user_books = similar_books_df['User-Book'].unique()
        other_books = similar_books_df['Book'].unique()
        all_rec_books = set(other_books) - set(user_books)  # Exclude user's books

        top_books = similar_books_df[similar_books_df['Book'].isin(all_rec_books)]
        top_books['Weighted-Similarity'] = pd.to_numeric(top_books['Weighted-Similarity'])
        top_books = top_books.drop_duplicates(subset=[self.bookid]).nlargest(top_n, 'Weighted-Similarity')
        return top_books[self.bookid].tolist()

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
            for user in tqdm(users[:50]):
                future = executor.submit(self._get_user_recommendations, user, k)
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

    df = pd.read_csv("data/prepared/preprocessed.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    model = ContentBasedFiltering()
    model.train(df_train)
    print(model.predict(df_test[model.userid].unique(), top_n=5))