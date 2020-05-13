'''
The most common library is spacy

Things to install
python3 -m pip install spacy
python3 -m spacy download en_core_web_sm -> this for english language module
'''

import spacy

'''
Spacy has language specific models.
For example to import model for english language
'''
nlp = spacy.load('en_core_web_sm')

'''
Now since the model is loaded we can insert any document to this model.
'''

my_document = "Tea is healthy and calming, don't you think?"

'''
We can process the above text using our nlp model we loaded
'''

doc = nlp(my_document)

#******************************** TOKENIZING ********************************************
'''
Tokenizing is the process of splitting the document in units/tokens(words and special chatacters in general, word level tokenizer)
It is a very important step before converting the words to vectors of numbers
'''

#Let's print all tokens in our document
for token in doc:
    print(token)

#******************************** TEXT PREPROCESSING ********************************************
print('******************************** TEXT PREPROCESSING ********************************************')
#1. Lemmantizing -> converting a number to its base form
#2. Stop words -> words that doesn't give much meaning to sentence
print(f"Token\t\t\tLemma\t\t\tStopword".format('Token','Lemma','Stopword'))
print('-'*40)
for token in doc:
    print(f"{str(token)}\t\t\t\t{token.lemma_}\t\t\t\t{token.is_stop}")

'''
removing lemmantizing and stop_words may help the model to focus on important keywords
but it can also make the prediction of model worse so they these are treated as hyperparameters
and tuned for data
'''

