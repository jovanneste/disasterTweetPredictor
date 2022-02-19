import pickle
import pandas as pd

test_features = pickle.load(open('test_features.sav', 'rb'))
test_labels = pickle.load(open('test_labels.sav', 'rb'))
logr_model = pickle.load(open('logr_model.sav', 'rb'))

target = logr_model.predict(test_features)



print(len(target), target)

df = pd.DataFrame({'target':target})

  
# saving the dataframe
df.to_csv('file1.csv')