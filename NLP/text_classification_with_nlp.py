import pandas as pd
spam = pd.read_csv('spam.csv')
print(spam)

# Bag of Words
'''
Suppose my vocabulary is having 10 uniique words -> {word1,word2,word3,...,word10}
And my text is like -> word1 word2 word3 word1 word4 word5 word10 word3
So vercotr representaion of my text using bag of words is like -> [2,1,2,1,1,0,0,0,0,1] corresponding to frequency of 
{word1,word2,word3,...,word10} respectively. Like in text word1,word3 came 2 times, word2,word4,word5,word10 came 1 time and
others came 0 times.
'''

#Term frequency - Inverse document frequency(TF-IDF)
'''
Bag of words is not proper representation of a document as compare to TF-IDF
Term Frequency(TF) = (no. of times a term t appeared in document)/ (length of document)
IDF = log_e(total no. of documents / no. of documents with term t)
TF-IDF(t) = TF * IDF

We can calculate TF-IDF score for all possible words for all documents.
Example:-
doc1 -> word1 Word2 Word1
doc2 -> word2 Word1 Word2
doc3 -> word2 word3 word2

IDF(Word1) = log_e(3/2) = .405
IDF(Word2) = log_e(3/3) = 0
IDF(Word3) = log_e(3/1) = 1.09

TFIDF {word1 word2 word3}

doc1 = [2/3 * IDF(word1) 1/3 * IDF(word2) 0] = [.27 0 0]
doc2 = [1/3 * .405 2/3 * 0 0] = [.135 0 0]
doc3 = [0 1/3 * 0 1/3 * 1.09] = [0 0 .363]
'''

#Building a Bag of words model
#TextCategorizer class converts text into bag of words and then implement simple linear model over it

'''
1. Create an empty model
2. Create TextCategorizer
3. Add TextCategorizer to empty model
'''

import spacy

#Step 1. Creatign blank model
nlp = spacy.blank('en')

#Step 2. Create TextCategorizer
textcat = nlp.create_pipe('textcat',config={"exclusive_classes":True,"architecture":'tfidf'})

'''
In the above code config is given two attributes.
1. exclusive_classes -> This means we have classification task and the categories do not overlap. For example -> spam,not spam etc.
2. architecture -> we have asked our categorizer to use 'Bag Of Words' model(bow)
'''

#Step3. Add categorizer to model
nlp.add_pipe(textcat)



#Step4. Since we have created our classifier now we need to tell what labels this classifier gonna classify
'''
In our case they are spam or ham
'''

textcat.add_label('ham')
textcat.add_label('spam')

'''
Now our classifier is ready to classify documents
'''

#Step5. Training

'''
The TextCategorizer requires data in following format
X = 'text document' list
Y = [{'cats':{label1 : True/False,label2 : True/False}}]
Y is a list for each text in X
'''

train_text = spam['text'].values
train_labels = [{'cats':{'ham': label == 'ham','spam': label == 'spam'}} for label in spam['label']]

#Step6. Combine the two in single list
train_data = list(zip(train_text,train_labels))

from spacy.util import minibatch
import random
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

losses = {}
for epoch in range(8):
    random.shuffle(train_data)
    batches = minibatch(train_data, size=2)
    for batch in batches:
        texts,labels = zip(*batch)
        nlp.update(texts,labels,sgd = optimizer,losses = losses)
    print(losses)

#Prediction
text_new = ['Had your mobile 11 months or more? U R entitle']
docs = [nlp.tokenizer(text) for text in text_new]

textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)

print(scores)

#For predicting labels further
predict_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predict_labels])