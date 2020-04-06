import scipy as sp
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os

data = pd.read_csv('dating-full.csv')
def rem_col(df):
    df['string_race'] = 0
    #df['count_race'] = 0
    if str(df['race'])[0] == str("\'") and str(df['race'])[-1] == str("\'"):
        df['string_race'] = str(df['race'])[1:-1]
     #   df['count_race'] = 1
    else:
        df['string_race'] = str(df['race'])
      #  df['count_race'] = 0

    df['string_race_o'] = 0    
    #df['count_race_o'] = 0
    if str(df['race_o'])[0] == str("\'") and str(df['race_o'])[-1] == str("\'"):
        df['string_race_o'] = str(df['race_o'])[1:-1]
     #   df['count_race_o'] = 1
    else:
        df['string_race_o'] = str(df['race_o'])
      #  df['count_race_o'] = 0

    df['string_field'] = 0    
    #df['count_field'] = 0
    if str(df['field'])[0] == str("\'") and str(df['field'])[-1] == str("\'"):
        df['string_field'] = str(df['field'])[1:-1]
     #   df['count_field'] = 1
    else:
        df['string_field'] = str(df['field'])
      #  df['count_field'] = 0
    return df
def mod_col(df):
    df['lower_field'] = 0
    #df['count_lower'] = 0
    if df['field'].islower() == False:
     #   df['count_lower'] = 1
        df['lower_field'] = str(df['field'].lower())
    else:
      #  df['count_lower'] = 0
        df['lower_field'] = str(df['field'])
    return df
def prune_gaming(a):    
    if a>10: 
        return 10
    else: 
        return a
def prune_reading(a): 
    if a>10: 
        return 10
    else: 
        return a
#axis =1 is same as axis = columns
t = data[['race','race_o','field']].apply(rem_col,axis=1)
data['race'] = t['string_race']
data['race_o'] = t['string_race_o']
data['field'] = t['string_field']

t1 = data[['field']].apply(mod_col,axis=1)
data['field'] = t1['lower_field']

t2 = data[['race','race_o','gender','field']]
for i in ['race','race_o','gender','field']:
    A = pd.DataFrame(np.sort(t2[i].unique()),columns=[i])
    A[i+'_enum'] = range(len(t2[i].unique()))
    t2 = t2.merge(A, on= i, how= 'left')
data['gender'] = t2['gender_enum']
data['race'] = t2['race_enum']
data['race_o'] = t2['race_o_enum']
data['field'] = t2['field_enum']
t3 = data[['attractive_important','sincere_important','intelligence_important',\
            'funny_important','ambition_important','shared_interests_important',\
            'pref_o_attractive','pref_o_sincere','pref_o_intelligence',\
            'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']]

t3['total'] = \
        t3[['attractive_important','sincere_important','intelligence_important',\
        'funny_important','ambition_important','shared_interests_important']].sum(axis=1)
for i in ['attractive_important','sincere_important','intelligence_important',\
        'funny_important', 'ambition_important',\
        'shared_interests_important']:            
    t3[i+"_new"] = t3[i].div(t3.total,axis="index")
t3['total1'] = \
        t3[['pref_o_attractive','pref_o_sincere','pref_o_intelligence',\
            'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']].sum(axis=1)
for i in ['pref_o_attractive','pref_o_sincere','pref_o_intelligence',\
        'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']:
    t3[i+"_new"] = t3[i].div(t3.total1,axis="index")
# In[ ]:
for i in ['attractive_important','sincere_important','intelligence_important',\
        'funny_important','ambition_important','shared_interests_important',\
        'pref_o_attractive','pref_o_sincere','pref_o_intelligence',\
        'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']:
    data[i] = t3[i+"_new"]
data['gaming'] = data['gaming'].map(prune_gaming)
data['reading'] = data['reading'].map(prune_reading)

labels = range(5)
list_col =  list(data.columns.values)
list_col_full = list(data.columns.values)
for i in ['gender','race','race_o','samerace','field','decision']:
    list_col.remove(i)
t4 = data[list_col_full] 
for i in list_col_full:
    if i not in list_col:
        t4[i+"_bin"] =  t4[i]
    else:
        t4[i+"_bin"] = pd.cut(t4[i], 5, labels=labels)  
for i in list_col:
    data[i] = t4[i+"_bin"]
data_test = t4.sample(random_state=25,frac=0.2)
#data_test.to_csv('testSet_.csv',index = False)
data_train = data[~data.index.isin(data_test.index)]
data_train.to_csv('trainingSet_nbc.csv',index = False)
