'''
@author : jhhalls

Fake News Detection

1. Import the libraries
2. Load the data
3. Handle the missing values
4. Split the data
5. Preprocess the text data
6. Fit the model
7. Predict the result
8. Visualize the results
'''



# import the libraries

import numpy as np
import pandas as pd 
import seaborn as sns
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

## load the data

train = pd.read_csv('../Fakenews_detector/data/train.csv')
test = pd.read_csv('../Fakenews_detector/data/test.csv')

# extract the labels
labels = train.label

# check for null values
null = train.isna().sum()

# drop the null values
train = train.dropna()

# split the data
x_train, x_test, y_train, y_test = train_test_split(train['text'],
                                                    labels,
                                                    test_size = 0.2, 
                                                    random_state = 7)

# check the shape of the splitted data
x_train.shape
y_train.shape
x_test.shape
y_test.shape

## Preprocess the Natural Language to vector format
# initialize the TfIdfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

#  fit and transform the data
tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)

#  predict the outcome
y_pred = pac.predict(tfidf_test)

# confusion matrix
cf = confusion_matrix(y_test, y_pred)

# Print the classification report
print(classification_report(y_test, y_pred))

# check the accuracy score
print('accuracy_score : ' , round(accuracy_score(y_test, y_pred)*100,3),  '%')

# plot the heatmap of the confustion matrix
sns.heatmap(cf, annot=True, fmt='d')

# Predict the unlabelled data
test_pred = pac.predict(tfidf_test)
test_output_df  = test_pred.to_DataFrame(test_pred)

# save the output
test_output_df.to_csv('test_prediciton')
