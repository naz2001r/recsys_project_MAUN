import numpy as np
import pandas as pd
from base_model import BaseModel

class CollaborativeFiltering(BaseModel):
    """Collaborative Filtering Algorithm"""
    MODEL_NAME = 'CollaborativeFiltering'

    def __init__(self, 
                 bookid: str = "ISBN", 
                 userid: str = "User-ID", 
                 bookrank: str = "Book-Rating") -> None:
        """
        Initialize the CollabFilter model.

        Args:
            bookid (str, optional):   The name of the column representing the book ID. Defaults to "ISBN".
            userid (str, optional):   The name of the column representing the user ID. Defaults to "User-ID".
            bookrank (str, optional): The name of the column representing the book rating. Defaults to "Book-Rating".
        """
        super().__init__(self.MODEL_NAME)

        self.bookid = bookid
        self.userid = userid
        self.bookrank = bookrank
        self.rank = None
        self.user_book_matrix = None
        self.similarities = None

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the CollabFilter model using the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the book ratings data.
        """
        print("Training model...")
        self.rank = df.groupby([self.bookid]).agg({
            self.bookrank: "mean"
        }).reset_index()

        print("Creating user-book matrix...")
        self.user_book_matrix = df.pivot_table(index=self.userid, columns=self.bookid, values=self.bookrank).fillna(0)

        print("Calculating similarities...")
        self.similarities = self.user_book_matrix.corr(method='pearson')
        print("Training complete.")

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

        for user in users:
            if user in self.user_book_matrix.index:
                # get the books the user has rated
                user_books = self.user_book_matrix.loc[user]
                user_books = user_books[user_books > 0].index.values

                # get the similar users
                similar_users = pd.Series()
                for book in user_books:
                    similar_users = similar_users.append(self.similarities[book].dropna())

                similar_users = similar_users.groupby(similar_users.index).sum()
                if user in similar_users:
                    similar_users = similar_users.drop(user, errors='ignore')
                similar_users = similar_users.sort_values(ascending=False)[:k]

                # get the books the similar users have rated
                if similar_users.index.values.size > 0:
                    predictions.append(similar_users.index.values)
                else:
                    # when we don't have any similar users, just recommend the top k books
                    print(f"No similar users found for user {user}. Recommending top {k} books.")
                    predictions.append([self.rank.nlargest(k, self.bookrank)[self.bookid].to_list()])
            else:
                # when we don't have any data on the user, just recommend the top k books
                print(f"User {user} not found in training data. Recommending top {k} books.")
                predictions.append([self.rank.nlargest(k, self.bookrank)[self.bookid].to_list()])

        return np.array(predictions)
    
if __name__ == "__main__":
    # Run this code to test if script is not failing
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/prepared/BX-Book-Ratings-Prepared.csv")[:10000]
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    model = CollaborativeFiltering()
    model.train(df_train)
    model.predict(df_test[model.userid].values, k=5)

