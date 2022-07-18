#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# In[6]:


#def get_features(meals):
    # meals = dataframe
        # get dummies of meals
        
        #pd.read_csv('E:\\graduation_project\\diet rec system\\nut_df.csv') # main dataset
#         nutrient_dummies = meals.nutrient.str.get_dummies(sep=' ')
#         disease_dummies = meals.Disease.str.get_dummies(sep=' ')
#         diet_dummies = meals.diet.str.get_dummies(sep=' ')
#         feature_df = pd.concat([nutrient_dummies,disease_dummies,diet_dummies],axis=1)
     
#         return feature_df


# In[ ]:


#get_features(dataset)


# In[ ]:


# def content_based(inputs):
#         df = pd.read_csv('E:\\graduation_project\\diet rec system\\nut_df.csv')
#         feature_df = get_features(df)
        
        
        
#         #initialize model with k=40 neighbors
#         model = NearestNeighbors(n_neighbors=40,algorithm='ball_tree')
        
#         # fit model with features
#         model.fit(feature_df)
        
#         # Empty dataframe to contain results
#         df_results = pd.DataFrame(columns=list(df.columns))
        
#         ####
#         total_features = list(feature_df.columns)
#         print(total_features)
#         d = dict()

#         for i in total_features:
#             d[i]= 0

#         for i in inputs:
#             d[i] = 1
        
#         final_input = list(d.values())    
#         print(d)

# ####
      
#         # getting distance and indices for k nearest neighbor
#         distnaces , indices = model.kneighbors([final_input])

#         for i in list(indices):
#             df_results = df_results.append(df.iloc[i])
            
            
# # convert zeros to ones for sample_input elements
                
#         df_results = df_results.filter(['name', 'nutrient', 'diet', 'Disease', 'ingredients', 'steps'])
#         df_results = df_results.drop_duplicates(subset=['name'])
#         df_results = df_results.reset_index(drop=True)
#         return df_results


# In[ ]:


# inputs = ['Fat', 'diabeties']
# content_based(inputs)


# In[ ]:


# def k_neighbor(inputs,feature_df,profiles,k):
        
#         #initialize model with k neighbors
#         model = NearestNeighbors(n_neighbors=k,algorithm='ball_tree')
        
#         # fit model with dataset features
#         model.fit(feature_df)
        
#         df_results = pd.DataFrame(columns=list(profiles.columns))
        
#         ####
#         total_features = list(feature_df.columns)
#         print(total_features)
#         d = dict()

#         for i in total_features:
#             d[i]= 0

#         for i in inputs:
#             d[i] = 1
        
#         final_input = list(d.values())    
#         print(d)

# ####
        
#         # getting distance and indices for k nearest neighbor
#         distnaces , indices = model.kneighbors([final_input])

#         for i in list(indices):
#             df_results = df_results.append(profiles.iloc[i])
#             df_results = df_results.filter(['user_id', 'nutrient', 'Disease', 'diet'])

#         df_results = df_results.reset_index(drop=True)
#         return df_results


# In[ ]:


# df = pd.read_csv('E:\\graduation_project\\diet rec system\\nut_df.csv')
# feature_df = get_features(profiles)
# k_neighbor(inputs,feature_df,profiles,10)


# In[ ]:


# def find_neighbors(profiles,user_features,k):
#         # dataframe = profiles
#         # features = user features

#         features_df = get_features(profiles)
#         total_features = features_df.columns  
#         d = dict()
#         for i in total_features:
#             d[i]= 0
#         for i in user_features:
#             d[i] = 1
#         final_input = list(d.values())
#         #rint(total_features)
        
#         similar_neighbors = k_neighbor([final_input],features_df,profiles,k)
#         return similar_neighbors


# In[ ]:


# find_neighbors(profiles, ['Fat', 'diabetes'],10)


# In[ ]:





# ## كل اللي قبل final recommender فكك منه

# # Final Recommender

# In[24]:


class Recommender:
    
    def __init__(self,profiles,recent_activity,meals):
        self.df = meals
        self.profiles = profiles
        self.recent_activity = recent_activity
    
    def get_features(self,dataframe):
        # get dummies of dataframe: meals or profiles
        nutrient_dummies = dataframe.nutrient.str.get_dummies(sep=' ')
        disease_dummies = dataframe.Disease.str.get_dummies(sep=' ')
        diet_dummies = dataframe.diet.str.get_dummies(sep=' ')
        feature_df = pd.concat([nutrient_dummies,disease_dummies,diet_dummies],axis=1)
     
        return feature_df
    
    
    
    def content_based(self, user_features):
        
        feature_df = self.get_features(self.df)
        
        #initialize model with k=40 neighbors
        model = NearestNeighbors(n_neighbors=40,algorithm='ball_tree')
        
        # fit model with features
        model.fit(feature_df)
        
        # Empty dataframe to contain results
        df_results = pd.DataFrame(columns=list(self.df.columns))
        
        ####
        total_features = list(feature_df.columns)
        #print(total_features)
        #print(user_features)
        d = dict()

        for i in total_features:
            d[i]= 0

        for i in list(user_features):
            d[i] = 1
        
        final_input = list(d.values())    
        #print(d)

####
      
        # getting distance and indices for k nearest neighbor
        distnaces , indices = model.kneighbors([final_input])

        for i in list(indices):
            df_results = df_results.append(self.df.iloc[i])
            
            
# convert zeros to ones for sample_input elements
                
        df_results = df_results.filter(['Meal_Id','name', 'nutrient', 'Disease', 'diet', 'ingredients', 'steps'])
        df_results = df_results.drop_duplicates(subset=['name'])
        df_results = df_results.reset_index(drop=True)
        return df_results
    
    
  #  def k_neighbor(self,user_features,dataframe,k):
  #      
  #      # THIS FUNCTION GETS THE KNNS FROM THE USER, 
  #      # KNNS MAY BE MEALS OR PROFILES
  #      # IN CASE OF MEALS, THE INPUT WILL BE USER USER_FEATURES, FEATURE_DF = DATAFRAME OF MEALS FEATURES AND DATAFRAME = MEALS
  #      # IN CASE OF PROFILES, THE INPUT WILL BE USER USER_FEATURES, FEATURE_DF = DATAFRAME OF PROFILES FEATURES AND DATAFRAME = PROFILES
  #      # BUT WE WILL KEEP THIS FOR CONTENT BASED AND MAKE find_neighbors FUNCTION FOR PROFILES
  #      
  #      feature_df = self.get_features(self.df)
  #      
  #      #initialize model with k neighbors
  #      model = NearestNeighbors(n_neighbors=k,algorithm='ball_tree')
  #      
  #      # fit model with dataset features
  #      model.fit(feature_df)
  #      
  #      ####
  #      total_features = list(feature_df.columns)
  #      print(total_features)
  #      d = dict()

   #     for i in total_features:
   #         d[i]= 0
#
#        for i in user_features:
#            d[i] = 1
#        
#        final_input = list(d.values())    
#        print(d)

####
        
#        df_results = pd.DataFrame(columns=list(dataframe.columns))
#        
        # getting distance and indices for k nearest neighbor
 #       distnaces , indices = model.kneighbors([final_input])

 #       for i in list(indices):
 #           df_results = df_results.append(dataframe.loc[i])

  #      df_results = df_results.reset_index(drop=True)
  #      return df_results
    
    
    
    def find_neighbors(self,user_features,k):
        # dataframe = profiles
        # features = user features
        features_df = self.get_features(self.profiles)
        
        #initialize model with k neighbors
        model = NearestNeighbors(n_neighbors=k,algorithm='ball_tree')
        
        # fit model with dataset features
        model.fit(features_df)
        
        
        total_features = features_df.columns  
        d = dict()
        for i in total_features:
            d[i]= 0
        for i in user_features:
            d[i] = 1
        final_input = list(d.values())
        #rint(total_features)
        
        similar_neighbors = pd.DataFrame(columns=list(self.profiles.columns))
        
        # getting distance and indices for k nearest neighbor
        distnaces , indices = model.kneighbors([final_input])

        for i in list(indices):
            similar_neighbors = similar_neighbors.append(self.profiles.loc[i])

        similar_neighbors = similar_neighbors.reset_index(drop=True)
        
      #  similar_neighbors = self.k_neighbor([final_input],dataframe,k)
        return similar_neighbors
    
    
    
    
    def user_based(self,user_features,user_id):
     
        
        similar_users = self.find_neighbors(user_features,10)
        users = list(similar_users.user_id)
    
        results = self.recent_activity[self.recent_activity.user_id.isin(users)] #taking acitivies
   
        results = results[results['user_id'] != user_id] # selecting those which are not reviewed by user
 
        meals = list(results.Meal_Id.unique())
      
        results = self.df[self.df.Meal_Id.isin(meals)]
    
        results = results.filter(['Meal_Id','name','nutrient','ingredients','steps'])

        results = results.drop_duplicates(subset=['name'])
        results = results.reset_index(drop=True)
        return results
    
    
        
    def recent_activity_based(self,user_id):
        recent_df = self.recent_activity[self.recent_activity['user_id'] == user_id]
        meal_ids = list(recent_df.Meal_Id.unique())
        recent_data = self.df[self.df.Meal_Id.isin(meal_ids)][['nutrient','category','Disease','diet']].reset_index(drop=True)

        disease = []
        diet = []
        nut = []
        user_features = []
        
        for i in range(recent_data.shape[0]):
            try:
                for word in recent_data.loc[i,'Disease'].split():
                    disease.append(word)
            except:
                pass
                
        for i in range(recent_data.shape[0]):
            try:
                for word in recent_data.loc[i,'diet'].split():
                    diet.append(word)
            except:
                pass
                
        for i in range(recent_data.shape[0]):
            for word in recent_data.loc[i,'nutrient'].split():
                nut.append(word)
                
        nut_counts = dict(Counter(nut))
        mean_nut = np.mean(list(nut_counts.values()))
        for i in nut_counts.items():
            if i[1] > mean_nut:
                user_features.append(i[0])
                
        #nut_counts = recent_data.nutrient.value_counts()
        #mean_nut = recent_data.nutrient.value_counts().mean()
        #features = list(nut_counts[recent_data.nutrient.value_counts() > mean_nut].index)
        
        
        dis_counts = dict(Counter(disease))
        mean_dis = np.mean(list(dis_counts.values()))
        for i in dis_counts.items():
            if i[1] > mean_dis:
                user_features.append(i[0])
        
        
        diet_counts = dict(Counter(diet))
        mean_diet = np.mean(list(diet_counts.values()))
        for i in diet_counts.items():
            if i[1] > mean_diet:
                user_features.append(i[0])
                
        similar_neighbors = self.find_neighbors(user_features,10)
        return similar_neighbors.filter(['Meal_Id','name','nutrient','ingredients','steps','rating'])
        
    def recommend(self,user_id):
        #finding user's profile features by id
        profile = self.profiles[self.profiles['user_id'] == user_id]
        user_features = []
        user_features.extend(profile['nutrient'].values[0].split())

        try:
            user_features.extend(profile['Disease'].values[0].split())
        except:
            pass
        try:
            user_features.extend(profile['diet'].values[0].split())
        except:
            pass
        
        feature_df = self.get_features(self.df)
        
        df0 = self.content_based(user_features)
        df1 = self.user_based(user_features,user_id)
        df2 = self.recent_activity_based(user_id)
        #df3 = self.k_neighbor(inputs,feature_df,dataframe,k)
        
        df = pd.concat([df0,df1,df2])
      
        df = df.drop_duplicates('ingredients').reset_index(drop=True)
        df = df[df['name'].isnull() == False].reset_index(drop=True)
        return df


# In[25]:


user_id = 'user_71'  # user id of current user

profiles = pd.read_csv('diet rec system\\user_profiles.csv') # profiles of all users
recent_activity = pd.read_csv('diet rec system\\recent_activity.csv') # recent activities of current user (meals liked,rated,searched,Purchased)
dataset = pd.read_csv('diet rec system\\nut_df.csv') # main dataset


example = Recommender(profiles,recent_activity,dataset)
result = example.recommend(user_id)
result


# In[26]:


result.isnull().sum()


# In[5]:


profiles = pd.read_csv('diet rec system\\user_profiles.csv') # profiles of all users
profiles.head()


# In[ ]:





# ## من أول هنا لحد الاخر فكك منه

# In[7]:


# profiles['nutrient'].values[2].split()


# In[8]:


# profiles.head()


# In[9]:


# recent_activity = pd.read_csv('diet rec system\\recent_activity.csv') # recent activities of current user (meals liked,rated,searched,Purchased)
# recent_activity.head()


# In[ ]:





# ## Check about user_71

# In[10]:


user_meal = pd.read_csv('diet rec system\\nut_df.csv') 
user_meal = user_meal[user_meal.columns[1:]]


# In[11]:


user_meal.head(3)


# In[11]:


# user_meal[user_meal.user_id == 'User_70']


# In[12]:


# profiles[profiles.user_id == 'user_70']


# In[27]:


import joblib


# In[28]:


joblib.dump(example,'Recommender_model1')


# In[29]:


model = joblib.load('Recommender_model1')


# In[30]:


model.recommend(user_id)


# In[ ]:




