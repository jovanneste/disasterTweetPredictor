import pandas as pd
import spacy 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression



def main():

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


	#train_data['clean_text'] = train_data['text'].apply(lambda x: text_pipeline_spacy(x))

	one_hot_vectorizer = CountVectorizer(tokenizer=text_pipeline_spacy, binary=True)
	train_features = one_hot_vectorizer.fit_transform(train_data['text'])
	train_labels = train_data['target']

	bayes_classifier = BernoulliNB()
	nb_model = bayes_classifier.fit(train_features, train_labels)
	print("Naive Bayes Classifier Accuracy: "+ str(nb_model.score(train_features, train_labels)))

	logr = LogisticRegression(solver='saga')
	logr_model = logr.fit(train_features, train_labels)
	print("Logistic Regression Classifier Accuracy: "+ str(logr_model.score(train_features, train_labels)))
	
#	logr_validation_predicted_labels = logr_model.predict(validation_features)
	

if __name__=='__main__':
	main()