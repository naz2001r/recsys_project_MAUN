
import glob
import os 
import sys
import pandas as pd

sys.path.append('src/models/')

from base_model import BaseModel

menu_options = {
    1: 'Predict with baseline model',
    2: 'Predict with collaborative filtering model',
    3: 'Predict with matrix factorization model',
    4: 'Predict with content based model',
    5: 'Predict with sentence transformer',
    6: 'Predict with hybrid nn model',
    7: 'Exit',
}

def print_book(item:pd.Series):
    print('========================')
    print(f'ISBN:                {item["ISBN"].values[0]}')
    print(f'Book-Title:          {item["Book-Title"].values[0]}')
    print(f'Book-Author:         {item["Book-Author"].values[0]}')
    print(f'Year-Of-Publication: {item["Year-Of-Publication"].values[0]}')
    print(f'Publisher:           {item["Publisher"].values[0]}')

def get_books_names(books: pd.DataFrame, predictions: list) -> list:
    return [books[books['ISBN'] == bookid] for bookid in predictions]

def get_model_file(model_name: str) -> str:
    list_of_files = glob.glob(f'./data/models/{model_name}/*.pkl', recursive=True)
    if list_of_files:
        if not os.path.isfile(list_of_files[0]):
            print('Model not found. Please pull the pipeline')
            return ''

    return list_of_files[0]

def print_menu():
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )

def predict_with_model(books: pd.DataFrame, loaded_models: dict, model_name: str, user: str, k: int) -> list:
    if not model_name in loaded_models.keys():
        model_file = get_model_file(model_name)
        if model_file:
            print(model_file)
            loaded_models[model_name] = BaseModel('base').load(model_file)
    
    if loaded_models[model_name]:
        print(f'Predicting for {user} user:')
        return get_books_names(books, loaded_models[model_name].predict([user], k)[0])

if __name__=='__main__':
    loaded_models = {}
    books = pd.read_csv('data/raw/BX-Books.csv', sep=';', encoding='ISO-8859-1', on_bad_lines='skip', low_memory=False)

    while(True):
        print_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        if option in [1, 2, 3, 4, 5, 6]:
            user = int(input('Enter desired userid from file: '))
            k = int(input('Enter desired number of books: '))

        if option == 1:
            predicted = predict_with_model(books, loaded_models, 'baseline', user, k)
            [print_book(item) for item in predicted]
        elif option == 2:
            predicted = predict_with_model(books, loaded_models, 'CollaborativeFiltering', user, k)
            [print_book(item) for item in predicted]
        elif option == 3:
            predicted = predict_with_model(books, loaded_models, 'MatrixFactorization', user, k)
            [print_book(item) for item in predicted]
        elif option == 4:
            predicted = predict_with_model(books, loaded_models, 'ContentBasedFiltering', user, k)
            [print_book(item) for item in predicted]
        elif option == 5:
            predicted = predict_with_model(books, loaded_models, 'STransformContentBase', user, k)
            [print_book(item) for item in predicted]
        elif option == 6:
            predicted = predict_with_model(books, loaded_models, 'HybridNNmodel', user, k)
            [print_book(item) for item in predicted]
            
        elif option == 7:
            print('Thanks for using!')
            exit()
        else:
            print('Invalid option. Please enter a number between 1 and 6.')