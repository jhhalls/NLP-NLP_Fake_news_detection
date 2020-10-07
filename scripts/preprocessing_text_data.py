

# import the libraries
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
%matplotlib inline

import seaborn as sns
import numpy as np # linear algebra
import pandas as pd #data processing

import os
import re
import nltk



# load the data
train=pd.read_csv('/fake-news/train.csv')
test=pd.read_csv('/fake-news/test.csv')


# split the reviews
real = train[train['label']==1]
fake = train[train['label']==0]

# ITERATE THROUGH THE DATA
## shape and size
def shape_size(df):
    shape = (df.shape) 
    size = (df.shape)
    combined = ('shape: {}\n','size: {}'.format(shape,size))
    return combined


# fill null values with whitespace
def fill_na_with_space(df):
    df=df.fillna(' ')
    return df


# merge the headline, author and news
def total_text(df):
    df['total']=df['title']+' '+df['author']+df['text']  
    print ('Updated Data:\n')
    return df



def real_words(df):


    real_words = ''
    fake_words = ''
    stopwords = set(STOPWORDS) 

    for val in df[df['label']==1].total: 

            # split the value 
            tokens = val.split() 

            # Converts each token into lowercase 
            for i in range(len(tokens)): 
                tokens[i] = tokens[i].lower() 

            real_words += " ".join(tokens)+" "

            return real_words
    
# create the list of words    
def wordcloud(df):    
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(real_words) 
    pass



# plot the graph by passing wordcloud as argument
def plot_graph(sorted_words)
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    return plt.show() 



# remove the puntuation using regex
def remove_punctuation(df):
    for key, value in df.iterrows():
#     print(key,value[-1], '\n'*4)
        sentence = value[-1]
        sub = re.sub(r'[^\w\s]','',sentence)
        print(sub)
        
        
# tokenize
def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens



# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

for review in positive_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

print("len(word_index_map):", len(word_index_map))



# list of stopwords
def my_stopwords():
    stop_words = stopwords.words('english')
    return stop_words


# lemmatization
def my_lemmatizer(word):
    lemmatizer=WordNetLemmatizer()    # initialize the lemmatizer
    input_str  = my_tokenizer(word)  #tokenize each word
    for word in input_str:     # iterate throught the tokenized words
        print(lemmatizer.lemmatize(word))  # return the lemmatized words

        
        
        
lemmatizer=WordNetLemmatizer()
for index,row in train.iterrows():
    filter_sentence = ''
    
    sentence = row['total']
    sentence = re.sub(r'[^\w\s]','',sentence) #cleaning
    words = nltk.word_tokenize(sentence) #tokenization
    words = [w for w in words if not w in stop_words]  #stopwords removal
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
    train.loc[index,'total'] = filter_sentence
        

        
        
train = train['total', 'label']


# tf-idf vectorizer
def vectorize_text(features, max_features):
    vectorizer = TfidfVectorizer( stop_words='english',
                            decode_error='strict',
                            analyzer='word',
                            ngram_range=(1, 2),
                            max_features=max_features
                            #max_df=0.5 # Verwendet im ML-Kurs unter Preprocessing                   
                            )
    feature_vec = vectorizer.fit_transform(features)
    return feature_vec.toarray()



# feature extraction using count-vectorize and tf-idf vectorizer
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
        
