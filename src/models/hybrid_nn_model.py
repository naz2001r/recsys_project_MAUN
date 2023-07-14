import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from base_model import BaseModel
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent
import time

from sentence_transformers import SentenceTransformer
tqdm.pandas()


def get_batch(dataset, batch_size, shuffle=True):
    if shuffle:
        dataset = dataset.sample(frac=1).reset_index(drop=True)
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

    def forward(self, user_ids, product_ids, features, device):
        user_embeds = self.user_embedding(user_ids)
        product_embeds = self.product_embedding(product_ids)
        feature_embeds = self.features_embedding.encode(features, convert_to_tensor=True, device=device)
        
        x = torch.cat((user_embeds, product_embeds, feature_embeds), dim=1).to(device)
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
                 patience: int = 5,
                 min_delta: float = 0.001,
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
        self.patience = patience
        self.min_delta = min_delta

        self.df = None
        self.rank = None
        self.model = None
        self.unique_users = None
        self.user_to_index = None
        self.book_to_index = None
        self.top_books = None

    def _get_device(self) -> str:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _train_nn(self, df: pd.DataFrame, val_df: pd.DataFrame, num_epoch: int = 100, device: str = 'cpu') -> HybridModel:
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
        self.book_to_index = pd.Series(data = range(len(df[self.bookid].unique())), index = df[self.bookid].unique())
        # self.book_to_index = {original: index for index, original in enumerate(df[self.bookid].unique())}

        # Add the indices to the dataframe
        df["user_index"] = df[self.userid].apply(lambda x: self.user_to_index[x])
        df["book_index"] = self.book_to_index[df[self.bookid]]
#        df["book_index"] = df[self.bookid].apply(lambda x: self.book_to_index[x])
        
        val_df["user_index"] = val_df[self.userid].apply(lambda x: self.user_to_index[x])
        val_df["book_index"] = val_df[self.bookid].apply(lambda x: self.book_to_index[x])

        # Create the model
        model = HybridModel(num_users, num_books, self.transformer_model)
        model = model.to(device=device)

        # Define the loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Early stopping variablessss
        min_validation_loss = np.inf
        counter = 0

        # Train the model
        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, data in enumerate(get_batch(df.copy(), batch_size=128, shuffle=True)):
                # Get the inputs
                user_ids = torch.LongTensor(data["user_index"].values).to(device=device)
                product_ids = torch.LongTensor(data["book_index"].values).to(device=device)
                features = data[self.titleid].tolist()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = model(user_ids, product_ids, features, device)
                loss = loss_function(outputs, torch.Tensor(data[self.bookrank].tolist()).to(device=device))
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
                user_ids =  torch.LongTensor(data["user_index"].values).to(device=device)
                product_ids = torch.LongTensor(data["book_index"].values).to(device=device)
                features = data[self.titleid].tolist()

                #Forward + backward
                outputs = model(user_ids, product_ids, features, device)
                loss = loss_function(outputs, torch.Tensor(data[self.bookrank].tolist()).to(device=device))
                
                # Print statistics
                valid_loss += loss.item()
            valid_loss = valid_loss / (i+1)
                
            print(f"[{epoch + 1}] Validation loss: {valid_loss}")

            # Early stopping check
            if valid_loss < min_validation_loss:
                min_validation_loss = valid_loss
                counter = 0
            elif valid_loss > (min_validation_loss + self.min_delta):
                counter += 1
                if counter >= self.patience:
                    print('Early stopping!\n')
                    return model

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

        self.top_books = self.rank.sort_values(by="prediction", ascending=False)[self.bookid].tolist()

        print("Computing unique users...")
        self.unique_users = df[self.userid].unique()

        device = self._get_device()
        print(f'Device selected {device}')

        print("Training model...")
        self.model = self._train_nn(df.copy(), val_df.copy(), num_epoch=self.num_epochs, device=device)

    def _get_user_predictions(self, user_id: str, top_n: int = 3, device: str = 'cpu') -> list:
        """
        Get the predictions for a single user.

        Args:
            user_id (str): The user ID.
            top_n (int, optional): The number of predictions to return. Defaults to 3.
            device (str, optional): The device to run the computations on. Defaults to 'cpu'.

        Returns:
            list: The top n predictions for the user.
        """

        pred_start_time = time.time()

        start_time = time.time()
        # If the user is not in the dataset, return the top n books
        if user_id not in self.unique_users:
            return self.top_books[:top_n]
        
        stop_time = time.time()
        print(f'self._get_top_n_books(top_n) = {stop_time - start_time}')
        
        start_time = time.time()
        # Get the user index
        user_index = self.user_to_index[user_id]
        
        stop_time = time.time()
        print(f'self.user_to_index[user_id] = {stop_time - start_time}')

        start_time = time.time()
        # Get the user's books
        user_books = self.df[self.df[self.userid] == user_id][self.bookid].unique()
        
        stop_time = time.time()
        print(f'self.df[self.df[self.userid] == user_id][self.bookid].unique() = {stop_time - start_time}')

        start_time = time.time()
        # Create a dataframe with all the books which the user has not read
        not_readed_books = list(set(self.df[self.bookid]) - set(user_books))
        stop_time = time.time()
        print(f'list(set(self.df[self.bookid]) - set(user_books)) = {stop_time - start_time}')

        if not not_readed_books:
            return self.top_books[:top_n]
        
        start_time = time.time()
        all_books = pd.DataFrame({self.bookid: not_readed_books})
        
        stop_time = time.time()
        print(f'all_books = pd.DataFrame({self.bookid: not_readed_books}) = {stop_time - start_time}')

        # Add the user index
        all_books["user_index"] = user_index

        start_time = time.time()
        # Add the book index
        # all_books["book_index"] = all_books[self.bookid].apply(lambda x: self.book_to_index[x])
        all_books["book_index"] = self.book_to_index[all_books[self.bookid]]

        stop_time = time.time()
        print(f'all_books["book_index"] = self.book_to_index[all_books[self.bookid]] = {stop_time - start_time}')
        
        # Add the book title
        start_time = time.time()
        all_books[self.titleid] = all_books[self.bookid].apply(lambda x: self.df[self.df[self.bookid] == x][self.titleid].iloc[0])

        stop_time = time.time()
        print(f'all_books[self.bookid].apply(lambda x: self.df[self.df[self.bookid] == x][self.titleid].iloc[0]) = {stop_time - start_time}')
        
        pred_stop_time = time.time()
        print(f'Pandas part finished {pred_stop_time - pred_start_time}')

        start_time = time.time()
        # Get the predictions
        user_index_tensor = torch.LongTensor(all_books["user_index"].values).to(device)
        book_index_tensor = torch.LongTensor(all_books["book_index"].values).to(device)

        with torch.no_grad():
            predictions = self.model(user_index_tensor, book_index_tensor, all_books[self.titleid].tolist(), device)
        all_books["prediction"] = predictions.detach().cpu().numpy()

        stop_time = time.time()
        print(f'NN part used time = {stop_time - start_time}')

        stop_time = time.time()
        # Sort the books by their predicted rank
        all_books = all_books.sort_values(by="prediction", ascending=False)
        prediction = all_books[:top_n][self.bookid].tolist()
        if len(prediction) != top_n:
            n_missing = top_n - len(prediction)
            prediction.extend(self.top_books[:n_missing])
            print(f'Books appended for user {user_id}')

        stop_time = time.time()
        print(f'Post processing part used time = {stop_time - start_time}')
        return prediction

    def predict(self, users: str, k: int = 3) -> np.array:
        """
        Get the predictions for a list of users.

        Args:
            users (str): The list of users to get the predictions for.
            k (int, optional): The number of predictions to return. Defaults to 3.
        
        Returns:
            np.array: The predictions for the users.
        """
        device = self._get_device()
        self.model = self.model.to(device=device)
        print(f'Device selected {device}')

        predictions = []
        # with ThreadPoolExecutor(max_workers=1) as executor:
        #     futures = []
        #     print("Loading executor")
        #     for user in tqdm(users):
        #         future = executor.submit(self._get_user_predictions, user, k, device)
        #         futures.append(future)
            
        #     print("Collecting results from executor")
        #     with tqdm(total=len(users)) as pbar:
        #         for future in concurrent.futures.as_completed(futures):
        #             result = future.result()
        #             predictions.append(result)
        #             pbar.update(1)

        for i in tqdm(range(len(users))):
            predictions.append(self._get_user_predictions(users[i], k, device))

        return np.array(predictions)
