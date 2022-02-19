import pickle

test_features = pickle.load(open('test_features.sav', 'rb'))
logr_model = pickle.load(open('logr_model.sav', 'rb'))

logr_test_predicted_labels = logr_model.predict(test_features)
print(len(logr_test_predicted_labels), logr_test_predicted_labels)