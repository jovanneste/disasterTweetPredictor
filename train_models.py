import pandas as pd
import spacy 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

import pickle


print("Loading data...")

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

random_training_data = train_data.sample(frac=1)

review_limit = min(400000, len(random_training_data))
random_training_data = random_training_data.iloc[:review_limit, :]

train_split = int(len(random_training_data)*0.8)
train_data = random_training_data.iloc[:train_split, :]
validation_data = random_training_data.iloc[train_split:, :]

print('Training set contains {:d} reviews.'.format(len(train_data)))
print('Vadlidation set contains {:d} reviews.'.format(len(validation_data)))
#print('Test set contains {:d} reviews.'.format(len(test_data)))

number_positive_train = sum(train_data['target'] == 1)
number_positive_validation = sum(validation_data['target'] == 1)

print('Training set contains %0.0f%% positive reviews' % (100*number_positive_train/len(train_data)))
print('Validation set contains %0.0f%% positive reviews' % (100*number_positive_validation/len(validation_data)))




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


print("Training models...")
one_hot_vectorizer = CountVectorizer(tokenizer=text_pipeline_spacy, binary=True)

train_features = one_hot_vectorizer.fit_transform(train_data['text'])
train_labels = train_data['target']

test_features = one_hot_vectorizer.transform(test_data['text'])

validation_features = one_hot_vectorizer.transform(validation_data['text'])


pickle.dump(validation_data, open('validation_data.sav', 'wb'))
pickle.dump(validation_features, open('validation_features.sav', 'wb'))

pickle.dump(test_features, open('test_features.sav', 'wb'))
pickle.dump(test_data, open('test_labels.sav', 'wb'))



bayes_classifier = BernoulliNB()
nb_model = bayes_classifier.fit(train_features, train_labels)
pickle.dump(nb_model, open('nb_model.sav', 'wb'))

logr = LogisticRegression(solver='saga')
logr_model = logr.fit(train_features, train_labels)
pickle.dump(logr_model, open('logr_model.sav', 'wb'))




	

