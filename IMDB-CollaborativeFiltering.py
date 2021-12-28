#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Reading the main dataset
movies_df = pd.read_csv('movies.csv')
movies_df


# In[3]:


#Reading the reviews
ratings_df = pd.read_csv('ratings_sample.csv')
ratings_df


# In[4]:


#Extracting the year from the title column and mention them in a seperated column:
movies_df ['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))' , expand = False)
movies_df ['year'] = movies_df.year.str.extract('(\d\d\d\d)' , expand = False)
movies_df ['title']= movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df ['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df


# In[6]:


#Since this method is based on the other's preferances, genes are not important to us, so we can drop them:
movies_df = movies_df.drop('genres', axis = 1)
movies_df


# In[7]:


#The colums of 'timestamp' in the dataset of rating also are useless to us, so let's drop it:
ratings_df = ratings_df.drop('timestamp',axis = 1)
ratings_df


# In[8]:


#Now, we should make a dataframe from the input of the assumed user:
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':'Pulp Fiction', 'rating':5},
            {'title':'Akira', 'rating':4.5}
]
inputMovies = pd.DataFrame(userInput)
inputMovies


# In[10]:


# Now, we should extract the 'movieId' of the user's films through the main dataset:
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

#Then merge the 'movieId' to user dataframe:
inputMovies = pd.merge(inputId,inputMovies)
inputMovies


# In[11]:


#It seems that 'year' would be useless for us. So, let's drop it:
inputMovies = inputMovies.drop('year',axis = 1)
inputMovies


# In[12]:


#Now we should see what are the scores of the user's films in the main dataset:
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset


# In[17]:


#now, we can differentiate the userSubset dataframe by 'userId':
userSubsetGroup = userSubset.groupby('userId')
userSubsetGroup.get_group(42118)


# In[18]:


#Now we should prioritize the 'userId's that have the most mutual film with the assumed user:
userSubsetGroup = sorted(userSubsetGroup , key = lambda x:len(x[1]) , reverse = True)


# In[20]:


userSubsetGroup [0:5]


# In[21]:


#Now, we can get a reasonable number of userId with the mutual rating. I imagine 75 people:
userSubsetGroup = userSubsetGroup [0:75]


# In[22]:


#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0


# In[26]:


# To make the Pearson a DataFrame
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient = 'index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF


# In[27]:


# Noe we can sort the users:
topUsers = pearsonDF.sort_values(by = 'similarityIndex', ascending = False) 
topUsers


# In[28]:


# Here we add some other infromation about the users from the main dataset:
topUsersRating = topUsers.merge(ratings_df, left_on= 'userId' , right_on = 'userId', how = 'inner')
topUsersRating


# In[29]:


topUsersRating ['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating
#wightedRating is the approxiamtion of the rating by our user


# In[30]:


#Applies the sum to the topUser after grouping it up by userId
tempTopUserRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
tempTopUserRating.columns = ['sum_similarityIndex' , 'sum_weightedRating']
tempTopUserRating


# In[31]:


#Create an Empty DataFrame
recommendation_df = pd.DataFrame()

#Now we take the weighted Average
recommendation_df ['weighted Average Recommendation Score'] = tempTopUserRating['sum_weightedRating']/tempTopUserRating['sum_similarityIndex']
recommendation_df ['movieId'] = tempTopUserRating.index
recommendation_df   


# In[35]:


# now, we can sot the Values according to the 'weighted Average Recommendation Score'
Final_Rec_Sys = recommendation_df.sort_values(by= 'weighted Average Recommendation Score')
Final_Rec_Sys


# In[36]:


movies_df.loc[movies_df['movieId'].isin(Final_Rec_Sys.head(10)['movieId'].tolist())]


# In[ ]:




