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

train, test = train_test_split(df, test_size = 0.2, stratify = df['label'], random_state=3)


train.shape, test.shape

# create a TF-IDF vectorizer object
tfidf_vectorizer = TfidfVectorizer(lowercase= True, max_features=1000, stop_words='english')

# fit the object with the training data tweets
tfidf_vectorizer.fit(train.tweet)


train_idf = tfidf_vectorizer.transform(train.tweet)
test_idf  = tfidf_vectorizer.transform(test.tweet)


model_LR = LogisticRegression()


model_LR.fit(train_idf, train.label)


predict_train = model_LR.predict(train_idf)


predict_test = model_LR.predict(test_idf)

# define the stages of the pipeline
pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
                                                      max_features=1000,
                                                      stop_words= 'english')),
                            ('model', LogisticRegression())])


pipeline.fit(train.tweet, train.label)

# sample tweet
text = ["Excited to announce our latest product launch! Stay tuned for updates and get ready to elevate your experience. #productlaunch #innovation #excited"]


pipeline.predict(text)
