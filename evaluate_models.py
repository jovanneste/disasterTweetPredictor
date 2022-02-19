import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score


def main():
	nb_model = pickle.load(open('nb_model.sav', 'rb'))
	train_features = pickle.load(open('train_features.sav', 'rb'))
	train_labels = pickle.load(open('train_labels.sav', 'rb'))

	logr_model = pickle.load(open('logr_model.sav', 'rb'))
	

	print("\nNaive Bayes Classifier Accuracy: "+ str(nb_model.score(train_features, train_labels)))
	print("Logistic Regression Classifier Accuracy: "+ str(logr_model.score(train_features, train_labels))+"\n")



	def evaluation_summary(description, true_labels, predictions, target_classes=["0","1"]):
		print("Evaluation for: " + description)
		print(classification_report(true_labels, predictions,  digits=3, zero_division=0, target_names=target_classes))
		print('\nConfusion matrix:\n',confusion_matrix(true_labels, predictions)) # Note the order here is true, predicted

	NB_validation_predicted_labels = nb_model.predict(train_features)
	evaluation_summary("One-hot Naive Bayes Model",  train_labels, NB_validation_predicted_labels,  ["negative","positive"])

	LOGR_validation_predicted_labels = logr_model.predict(train_features)
	evaluation_summary("Logistic Regression Model",  train_labels, LOGR_validation_predicted_labels,  ["negative","positive"])


if __name__ == '__main__':
	main()