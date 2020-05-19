'''
WordEmbeddings -> better way to represent words into vectors.
WordEmbeddings are learned through ML model such that the words used in same context have similar vector representation.
For example words like -> lion, tiger will be closer than planet, castle
Also they mathmatically compliments.Like
Vec(Male) - Vec(Female) = Vec(King) - Vec(Queen)

here left hand side and right hand side both just differs on gender and that feature is perfectly captured by WordEmbeddings
'''

'''
There are many word embeddings. the 2 most talked about embeddings are :-
1. Word2Vec
2. Bert
'''

import numpy as np
import spacy

nlp = spacy.load('en_core_web_lg') #for Word2Vec

test_text = "Hi! this is test text"
with nlp.disable_pipes(): #Disabling other features
    vectors = np.array([token.vector for token in nlp(test_text)])

print(vectors.shape)

'''
The above is for word level embeddings but not for doc level embeddings.
The simplest way is average of all words in a document to create doc level embedding
'''

import pandas as pd
spam = pd.read_csv('spam.csv')
with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector for text in spam.text])

print(doc_vectors.shape)


#Classification
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(doc_vectors,spam.label,test_size=0.1,random_state=1)

'''
One of the important classifier is SVM. Here we are using linear SVM.
'''

from sklearn.svm import LinearSVC
svc = LinearSVC(random_state=1,dual=False,max_iter=10000)
svc.fit(X_train,y_train)
print(f"Accuracy : {svc.score(X_test,y_test) * 100:.3f}%",)

#Document Similarity
def cosine_similarity(a,b):
    return a.dot(b)/np.sqrt(a.dot(a) * b.dot(b))

a = nlp("This is test task").vector
b = nlp("A task which can be used for test").vector

print(cosine_similarity(a,b))


'''
Need to download following model for using BERT
model name = en_trf_bertbaseuncased_lg
more on this after covering *Transformer Neural Network*
'''