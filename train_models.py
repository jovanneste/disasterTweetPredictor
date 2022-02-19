import pandas as pd
import spacy 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

import pickle



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

nlp = spacy.load('en_core_web_sm', disable=["ner"])
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')

def text_pipeline_spacy(text):
	tokens = []
	doc = nlp(text)
	for t in doc:
		if not t.is_stop and not t.is_punct and not t.is_space:
			tokens.append(t.lemma_.lower())
	return tokens



one_hot_vectorizer = CountVectorizer(tokenizer=text_pipeline_spacy, binary=True)
train_features = one_hot_vectorizer.fit_transform(train_data['text'])
train_labels = train_data['target']

test_features = one_hot_vectorizer.transform(test_data['text'])
pickle.dump(test_features, open('test_features.sav', 'wb'))


pickle.dump(train_features, open('train_features.sav', 'wb'))
pickle.dump(train_labels, open('train_labels.sav', 'wb'))


bayes_classifier = BernoulliNB()
nb_model = bayes_classifier.fit(train_features, train_labels)
pickle.dump(nb_model, open('nb_model.sav', 'wb'))

logr = LogisticRegression(solver='saga')
logr_model = logr.fit(train_features, train_labels)
pickle.dump(logr_model, open('logr_model.sav', 'wb'))




	

