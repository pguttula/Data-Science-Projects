import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re, string
import sys
import time
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
np.random.seed(2018)
from nltk.corpus import stopwords
nltk.download('wordnet')
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#Read preprocessed data from csv
news = pd.read_csv('Sentiment_business_labelled_new.csv',encoding='latin-1')
df =  news
df=df.dropna(subset=['Label'])
df = df[df['headline_text'].apply(lambda x: len(x) > 20)]
df = df.reset_index(drop=True)
df =  news[['preprocess','Label']]

#code to select equal number of positive, negative and neutral rows to make the dataset more balanced.
df_p=(df[df['Label']=="Positive"])
df_p = df_p.sample(n=450)
df_n=(df[df['Label']=="Negative"])
df_n = df_n.sample(n=450)
df_ne=(df[df['Label']=="Neutral"])
df = pd.concat([df_p,df_n,df_ne])
df['index'] = df.index
stemmer=PorterStemmer()
print("length of dataset after balancing the label counts: ",len(df))

#Replace data_classes with integer labels
data_classes = ['Positive',"Negative","Neutral"]
df['Label'] = df['Label'].apply(data_classes.index)

#TF-IDF Word embedding
tfidf = TfidfVectorizer(lowercase=False)
ml_data = tfidf.fit_transform(df['preprocess'])

#split training and test data
x_train, x_test, y_train, y_test = train_test_split(ml_data,df['Label'], test_size=0.2,random_state=1)

#LR implementation
model = LogisticRegression(solver='lbfgs',multi_class='auto', max_iter=1000)
model.fit(x_train,y_train)
predicted = model.predict(x_test)
print("LR score on Test dataset: {:.2f}".format(accuracy_score(y_test,predicted)))

#DT implementation
model = DecisionTreeClassifier(random_state=0)
model.fit(x_train,y_train)
predicted = model.predict(x_test)
print("DT score on Test dataset: {:.2f}".format(accuracy_score(y_test,predicted)))


#Random Forests implementation
classifier = rfc()
classifier.fit(x_train,y_train)
predicted = classifier.predict(x_test)
print("RF score on Test dataset: {:.2f}".format(accuracy_score(y_test,predicted)))

#svm implementation
classifier = SVC(gamma='auto')
classifier.fit(x_train,y_train)
predicted = classifier.predict(x_test)
print("SVM score on Test dataset: {:.2f}".format(accuracy_score(y_test,predicted)))

#Multi Layer Perceptron implementation
nn = MLPClassifier(solver='adam', activation='relu', random_state=1, batch_size = 128,
                   max_iter = 1000, learning_rate_init = 0.01,
                   hidden_layer_sizes=[512, 256, 128, 64, 28])
nn.fit(x_train, y_train)

print("MLP with RELU function", nn.score(x_test, y_test))
