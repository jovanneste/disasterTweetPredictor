import pickle
import pandas as pd

test_features = pickle.load(open('test_features.sav', 'rb'))
test_labels = pickle.load(open('test_labels.sav', 'rb'))
logr_model = pickle.load(open('logr_model.sav', 'rb'))
nb_model = pickle.load(open('nb_model.sav', 'rb'))
combined_model = pickle.load(open('combined_model.sav', 'rb'))
one_hot_train_features = pickle.load(open('one_hot_train_features.sav', 'rb'))

target = logr_model.predict(test_features)


print(str(len(target)) + " entries")

df = pd.DataFrame({'id':test_labels['id'], 'target':target})
df.reset_index(drop=True)

#print("Some random entries: \n")
#for i in range(0,50,2):
#	print(str(test_labels['text'][i]) + " -- prediction -- " + str(target[i]))

print("\nSaved to file predictions.csv")
# saving the dataframe
df.to_csv('predictions.csv', index=False)