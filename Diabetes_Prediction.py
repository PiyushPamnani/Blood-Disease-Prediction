import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("diabetes.csv")
# data = np.array(data)

data.groupby('Outcome').mean()

X = data.drop(columns='Outcome', axis=1)

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
Y = data['Outcome']
# print(X,y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)


pickle.dump(scaler, open('model2.pkl', 'wb'))
model2 = pickle.load(open('model2.pkl', 'rb'))

pickle.dump(classifier, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
