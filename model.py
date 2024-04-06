import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from joblib import dump

df = pd.read_csv('https://raw.githubusercontent.com/lakshay-arora/Hate-Speech-Classification-deployed-using-Flask/master/dataset/twitter_sentiments.csv')
df.head()
#df.duplicated().sum()

# @title label

from matplotlib import pyplot as plt
df['label'].plot(kind='hist', bins=20, title='label')
plt.gca().spines[['top', 'right',]].set_visible(False)

# train test split
train, test = train_test_split(df, test_size = 0.2, stratify = df['label'], random_state=3)

# get the shape of train and test split.
train.shape, test.shape

# create a TF-IDF vectorizer object
tfidf_vectorizer = TfidfVectorizer(lowercase= True, max_features=1000, stop_words='english')

# fit the object with the training data tweets
tfidf_vectorizer.fit(train.tweet)

# transform the train and test data
train_idf = tfidf_vectorizer.transform(train.tweet)
test_idf  = tfidf_vectorizer.transform(test.tweet)

# create the object of LinearRegression Model
model_LR = LogisticRegression()

# fit the model with the training data
model_LR.fit(train_idf, train.label)

# predict the label on the traning data
predict_train = model_LR.predict(train_idf)

# predict the model on the test data
predict_test = model_LR.predict(test_idf)

# define the stages of the pipeline
pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
                                                      max_features=1000,
                                                      stop_words= 'english')),
                            ('model', LogisticRegression())])

# fit the pipeline model with the training data
pipeline.fit(train.tweet, train.label)

# sample tweet
text = ["Excited to announce our latest product launch! Stay tuned for updates and get ready to elevate your experience. #productlaunch #innovation #excited"]

# predict the label using the pipeline
pipeline.predict(text)
