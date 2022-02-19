import pickle
import pandas as pd

test_features = pickle.load(open('test_features.sav', 'rb'))
test_labels = pickle.load(open('test_labels.sav', 'rb'))
logr_model = pickle.load(open('logr_model.sav', 'rb'))

target = logr_model.predict(test_features)



print(str(len(target)) + " entries")

df = pd.DataFrame({'target':target})

print("Some random entries: \n")
for i in range(0,50,2):
	print(str(test_labels['text'][i]) + " -- prediction -- " + str(target[i]))

print("\nSaved to file predictions.csv")
# saving the dataframe
df.to_csv('predictions.csv')