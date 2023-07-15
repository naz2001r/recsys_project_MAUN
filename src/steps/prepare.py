import os
import sys
import numpy as np
import pandas as pd

from constants import mapping_dict
pd.set_option('mode.chained_assignment', None)

import warnings
warnings.filterwarnings("ignore")
import dvc.api

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare.py input_dir output-dir\n'
    )
    sys.exit(1)

output_dir = sys.argv[2]

params = dvc.api.params_show()
rate_file = params['prepare']['rate_file']
users_file = params['prepare']['users_file']
book_file = params['prepare']['book_file']

os.makedirs(output_dir, exist_ok=True)

input_folder = sys.argv[1]
print('Read files')
books = pd.read_csv(os.path.join(input_folder, book_file), sep=';', encoding='ISO-8859-1', on_bad_lines='skip', low_memory=False)
users = pd.read_csv(os.path.join(input_folder, users_file), sep=';', encoding='ISO-8859-1', on_bad_lines='skip')
ratings = pd.read_csv(os.path.join(input_folder, rate_file), sep=';', encoding='ISO-8859-1', on_bad_lines='skip')

print('Start preprocessing books dataset')
# First work with books dataset

# Delete unseful columns Image-URL-S, Image-URL-M and Image-URL-L.
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
# Let's discover the year of publication. In our data we have strange years like '0', 'DK Publishing Inc', 'Gallimard' 
# and as dataset was publised in 2014 we should also remove years like 2020', '2021', '2024', '2026', '2030', '2037', '2038', '2050'
# investigating the rows having 'DK Publishing Inc' as yearOfPublication
# It is evident that there are inaccuracies in the Year-Of-Publication field. The dataset seems to have mistakenly recorded 'DK Publishing Inc' and 'Gallimard' as Year-Of-Publication entries due to errors in the CSV file.
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','Year-Of-Publication'] = 2000
books.loc[books.ISBN == '0789466953','Book-Author'] = "James Buckley"
books.loc[books.ISBN == '0789466953','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','Book-Title'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','Year-Of-Publication'] = 2000
books.loc[books.ISBN == '078946697X','Book-Author'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','Book-Title'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

# making required corrections for the rows having 'Gallimard' as yearOfPublication, keeping other fields intact
books.loc[books.ISBN == '2070426769','Year-Of-Publication'] = 2003
books.loc[books.ISBN == '2070426769','Book-Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','Publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','Book-Title'] = "Peuple du ciel, suivi de 'Les Bergers"

# Convert Year_of_Publication to int type
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
books.loc[(books['Year-Of-Publication'] > 2006) | (books['Year-Of-Publication'] == 0),'Year-Of-Publication'] = np.NAN

# replacing NaNs with median value of Year-Of-Publication
books['Year-Of-Publication'].fillna(round(books['Year-Of-Publication'].median()), inplace=True)

# exploring 'publisher' column were we have 2 nan and chage them to field 'other'
books.Publisher.fillna('other',inplace=True)
# exploring 'Book-Author' column and filling Nan of Book-Author with others
books['Book-Author'].fillna('other',inplace=True)

print('End preprocessing books dataset\n')

print('Start preprocessing ratings dataset')
# Work with ratings dataset

#  Ratings dataset should have books only which exist in our books dataset and should have ratings from users which exist in users dataset. 

ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
ratings_new = ratings_new[ratings_new['User-ID'].isin(users['User-ID'])]

# The ratings are very unevenly distributed, and the vast majority of ratings are 0 .
# As quoted in the description of the dataset - BX-Book-Ratings contains the book rating information. 
# Ratings are either explicit, expressed on a scale from 1-10 higher values denoting higher appreciation, or implicit, expressed by 0.
# Hence segragating implicit and explict ratings datasets.

ratings_explicit = ratings_new[ratings_new['Book-Rating'] != 0]
ratings_implicit = ratings_new[ratings_new['Book-Rating'] == 0]

# Create column Rating with mean value of every book rate 
ratings_explicit['Avg_Rating'] = ratings_explicit.groupby('ISBN')['Book-Rating'].transform('mean')

# Create column Total_No_Of_Users_Rated with total number of user who rate the book
ratings_explicit['Total_No_Of_Users_Rated'] = ratings_explicit.groupby('ISBN')['Book-Rating'].transform('count')

print('End preprocessing ratings dataset\n')

# Work with users dataset
print('Start preprocessing users dataset')

# In our data we can see that there are many users with age more than 100, and what is most unexpected that there are those whose age are 200. This is outliers so we should drop it. 
# Also, we should drop the users with age lower than 5, because children can start to read in this age.
users.loc[(users.Age > 100) | (users.Age < 5), 'Age'] = np.nan

# In column location we have 57339 unique Value and it's really hard to understand. So lets create column Country.

# Split 'Location' column into three separate columns
users[['City/Town', 'State/Province/Region', 'Country']] = users['Location'].str.split(', ', expand=True, n=2)

#drop location column
users.drop('Location',axis=1,inplace=True)
users['Country'] = users['Country'].astype('str')

# Replase unknown countries 
users['Country'].replace(mapping_dict, inplace=True)

# If it is region/state, country - take only country
users['Country'] = users['Country'].str.split(',').str[-1].str.strip()

# Remove Nan in Age by fill it with median value based on country
users['Age'] = users['Age'].fillna(users.groupby('Country')['Age'].transform('median'))

# Still we have 276 Nan values let's fill them with mean
users['Age'].fillna(users.Age.mean(),inplace=True)

print('End preprocessing users dataset\n')

# Merge all data and save to csv
dataset = users.copy()
dataset = pd.merge(dataset,ratings_explicit,on='User-ID')
dataset = pd.merge(dataset,books,on='ISBN')
dataset.to_csv(os.path.join(output_dir, 'preprocessed.csv'),index=False)
