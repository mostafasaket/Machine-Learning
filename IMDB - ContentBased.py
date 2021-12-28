#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


# In[2]:


#The Title, year, Genres and ID of the films
movies_df = pd.read_csv('movies.csv')
movies_df


# In[3]:


#The ratings of the films which according to the audiences view
ratings_df = pd.read_csv('ratings.csv')
ratings_df


# There are something wrong in the movie dataset. The year should be divided from the title and goes to its own column:
# 

# In[4]:


movies_df ['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))' , expand = False)
movies_df ['year'] = movies_df.year.str.extract('(\d\d\d\d)' , expand = False)
movies_df ['title']= movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df ['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df


# In[5]:


# Now, we should divide the genres with ',' while they are divided now by ''
movies_df ['genres'] = movies_df.genres.str.split('|')
movies_df ['genres'].values


# In[6]:


#Make a copy from the main dataset(movies_df) to work on them and extracts some other feature (genders) from it
gendersMovie_df = movies_df.copy()


# In[7]:


#Since The genres are in a List, we cannot use 'get_dummies' to divide the genres
#SO:
for index , row in movies_df.iterrows():
    for genre in row['genres']:
        gendersMovie_df.at[index,genre] = 1
#Above nested FOR is using to put '1' instead of each genres that are existing in main dataset     

#Below fill the empty fields with '0'
gendersMovie_df = gendersMovie_df.fillna(0)
gendersMovie_df.head(20)


# In[8]:


#Let's again call ratings list:
ratings_df = pd.read_csv('ratings.csv')
ratings_df

#Since the column of timestamp does not work for us, it is recommneded to drop it:
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df


# In[9]:


ratings_df.dtypes


# So Far I was working on the data cleansing and making mada usable.

# In[ ]:





# ### Now let's focus on the two different RecSys:

# # Content-Based RecSys

# In[10]:


# Assume that a person watch below-mentioned films and rated them followingly:

userInput = [
            {'title': 'Breakfast Club, The', 'rating': 5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':'Pulp Fiction', 'rating':5},
            {'title':'Akira', 'rating':4.5}
]

#Now we have to make a DataFram from the 5 mentioned films:
inputMovies = pd.DataFrame(userInput)
inputMovies


# In[11]:


#To realize the column of movieId, we have to look for the films in the main dataset. The first step is looking for mutual
#title of the films in both dataset (our 5-film dataset and main dataset).
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

#To know what are the ratings of the films, we have to include the ratings:
inputMovies = pd.merge(inputId , inputMovies)
inputMovies


# In[12]:


#Now, there are some columns that are useless for us. Let's drop them:
inputMovies = inputMovies.drop('genres', axis=1).drop('year', axis = 1)
inputMovies


# In[13]:


# Now, we should learn what are the Genres (content) of the user interesting films list:
userMovies = gendersMovie_df[gendersMovie_df['title'].isin(inputMovies['title'].tolist())]
userMovies

#Now we have the list of user interesting list with their genres


# In[14]:


#We need the genres only of the films:
userMovies = userMovies.reset_index(drop=True)
userGenerTable = userMovies.drop('movieId', axis = 1).drop('title', axis = 1).drop('genres', axis = 1).drop('year', axis = 1)
userGenerTable


# In[15]:


inputMovies = inputMovies.reset_index(drop= True)
inputMovies


# In[16]:


inputMovies ['rating']


# In[17]:


# Now we should know what genres, get what score according to the user's rating of the films:
userProfile = userGenerTable.transpose().dot(inputMovies['rating'])

userProfile


# above-mentioned numbers say that what kind of genres are compatible more with the user's appetite.

# In[26]:


genresTable = gendersMovie_df.set_index(gendersMovie_df['movieId'])
genresTable = genresTable.drop('title',axis = 1).drop('year', axis = 1).drop('movieId' , axis =1).drop('genres', axis = 1)
genresTable


# In[35]:


#In this Step the score of each film based on the 5 initial scores of the films will be indicated.
RecSysTable_df = ((genresTable*userProfile).sum(axis = 1))/(userProfile.sum())
RecSysTable_df


# In[36]:


#We can sort the movies based on their score:
Sorted_RecSys = RecSysTable_df.sort_values(ascending = False)


# In[38]:


#Now we can locate the name and information of the films through the 'movieId'
movies_df.loc[movies_df['movieId'].isin(Sorted_RecSys.head(10).keys())]


# In[ ]:




