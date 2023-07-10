import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from base_model import BaseModel
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent

from sentence_transformers import SentenceTransformer
tqdm.pandas()


def get_batch(dataset, batch_size, shuffle=True):
    if shuffle:
        dataset =dataset.sample(frac=1).reset_index(drop=True)
    for i in range(0, len(dataset), batch_size):
        if i + batch_size > len(dataset):
            yield dataset.iloc[i:]
        yield dataset.iloc[i:i+batch_size]


class HybridModel(nn.Module):
    def __init__(self, num_users: int, num_products: int, transformer_model: str):
        super(HybridModel, self).__init__()

        self.features_embedding = SentenceTransformer(transformer_model)
        embeddings_size = self.features_embedding.get_sentence_embedding_dimension()

        self.user_embedding = nn.Embedding(num_users, embeddings_size)
        self.product_embedding = nn.Embedding(num_products, embeddings_size)
        self.fc1 = nn.Linear(embeddings_size * 3, embeddings_size)
        self.fc2 = nn.Linear(embeddings_size, embeddings_size)
        self.fc3 = nn.Linear(embeddings_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_ids, product_ids, features):
        user_embeds = self.user_embedding(user_ids)
        product_embeds = self.product_embedding(product_ids)
        feature_embeds = self.features_embedding.encode(features, convert_to_tensor=True)
        
        x = torch.cat((user_embeds, product_embeds, feature_embeds), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class HybridNN_Recommender(BaseModel):
    """HybridNNmodel Algorithm"""
    MODEL_NAME = 'HybridNNmodel'

    def __init__(self, 
                 transformer_model: str = 'all-MiniLM-L6-v2',
                 num_epochs: int = 100,
                 bookid: str = "ISBN", 
                 userid: str = "User-ID", 
                 bookrank: str = "Book-Rating",
                 titleid: str = "Book-Title") -> None:
        """
        Initialize the HybridNNmodel model.

        Args:
            transformer_model (str, optional): The name of the transformer model to use. Defaults to 'all-MiniLM-L6-v2'.
            num_epochs (int, optional): The number of epochs to train the model. Defaults to 100.
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
        self.transformer_model = transformer_model
        self.num_epochs = num_epochs

        self.df = None
        self.rank = None
        self.model = None
        self.unique_users = None
        self.user_to_index = None
        self.book_to_index = None

    def _train_nn(self, df: pd.DataFrame, val_df: pd.DataFrame, num_epoch: int = 100) -> HybridModel:
        """
        Train the NN model.

        Args:
            df (pd.DataFrame): The dataframe to train the model on.

        Returns:
            HybridModel: The trained model.
        """
        # Get the number of unique users and books
        num_users = len(self.unique_users)
        num_books = len(df[self.bookid].unique())

        # Create a dictionary mapping users and books to their indices
        self.user_to_index = {original: index for index, original in enumerate(self.unique_users)}
        self.book_to_index = {original: index for index, original in enumerate(df[self.bookid].unique())}

        # Add the indices to the dataframe
        df["user_index"] = df[self.userid].apply(lambda x: self.user_to_index[x])
        df["book_index"] = df[self.bookid].apply(lambda x: self.book_to_index[x])

        # Create the model
        model = HybridModel(num_users, num_books, self.transformer_model)

        # Define the loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, data in enumerate(get_batch(df.copy(), batch_size=128, shuffle=True)):
                # Get the inputs
                user_ids =  torch.LongTensor(data["user_index"].values)
                product_ids = torch.LongTensor(data["book_index"].values)
                features = data[self.titleid].tolist()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = model(user_ids, product_ids, features)
                loss = loss_function(outputs, torch.Tensor(data[self.bookrank].tolist()))
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f"[{epoch + 1}, {i + 1}] Training loss: {running_loss / 100}")
                    running_loss = 0.0

            valid_loss = 0.0
            for i, data in enumerate(get_batch(val_df.copy(), batch_size=128, shuffle=True)):
                # Get the inputs
                user_ids =  torch.LongTensor(data["user_index"].values)
                product_ids = torch.LongTensor(data["book_index"].values)
                features = data[self.titleid].tolist()

                #Forward + backward
                outputs = model(user_ids, product_ids, features)
                loss = loss_function(outputs, torch.Tensor(data[self.bookrank].tolist()))
                
                # Print statistics
                valid_loss += loss.item()
                
            print(f"[{epoch + 1}] Validation loss: {running_loss / (i+1) }")

            

        return model

    def train(self, df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """
        Train the model.

        Args:
            df (pd.DataFrame): The dataframe to train the model on.
        """
        print("Training model...")
        self.df = df

        print("Computing book rank...")
        self.rank = df.groupby([self.bookid]).agg({
            self.bookrank: "mean"
        }).reset_index()

        print("Computing unique users...")
        self.unique_users = df[self.userid].unique()

        print("Training model...")
        self.model = self._train_nn(df.copy(), val_df.copy(), num_epoch=self.num_epochs)


    def _get_user_predictions(self, user_id: str, top_n: int = 3) -> list:
        """
        Get the predictions for a single user.

        Args:
            user_id (str): The user ID.
            top_n (int, optional): The number of predictions to return. Defaults to 3.

        Returns:
            list: The top n predictions for the user.
        """

        # If the user is not in the dataset, return the top n books
        if user_id not in self.unique_users:
            return self.rank.nlargest(top_n, self.bookrank)[self.bookid].tolist()
        
        # Get the user index
        user_index = self.user_to_index[user_id]

        # Get the user's books
        user_books = self.df[self.df[self.userid] == user_id][self.bookid].unique()

        # Create a dataframe with all the books
        all_books = pd.DataFrame({self.bookid: self.df[self.bookid].unique()})

        # Add the user index
        all_books["user_index"] = user_index

        # Add the book index
        all_books["book_index"] = all_books[self.bookid].apply(lambda x: self.book_to_index[x])

        # Add the book title
        all_books[self.titleid] = all_books[self.bookid].apply(lambda x: self.df[self.df[self.bookid] == x][self.titleid].iloc[0])

        # Get the predictions
        all_books["prediction"] = self.model(
            torch.LongTensor(all_books["user_index"].values),
            torch.LongTensor(all_books["book_index"].values),
            all_books[self.titleid].tolist()
        ).detach().numpy()

        # Remove the books the user has already read
        all_books = all_books[~all_books[self.bookid].isin(user_books)]

        # Sort the books by their predicted rank
        all_books = all_books.sort_values(by="prediction", ascending=False)

        return all_books[:top_n][self.bookid].tolist()

    def predict(self, users: str, k: int = 3) -> np.array:
        """
        Get the predictions for a list of users.

        Args:
            users (str): The list of users to get the predictions for.
            k (int, optional): The number of predictions to return. Defaults to 3.
        
        Returns:
            np.array: The predictions for the users.
        """

        predictions = []
        with ThreadPoolExecutor() as executor:
            futures = []
            print("Loading executor")
            for user in tqdm(users):
                future = executor.submit(self._get_user_predictions, user, k)
                futures.append(future)
            
            print("Collecting results from executor")
            with tqdm(total=len(users)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    predictions.append(result)
                    pbar.update(1)
        return np.array(predictions)
