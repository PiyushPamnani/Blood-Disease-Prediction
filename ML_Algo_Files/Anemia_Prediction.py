import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv("anemia.csv")

data.groupby('Result').mean()

X = data.drop(columns='Result', axis=1)

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
Y = data['Result']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

pickle.dump(scaler, open('anemia_model2.pkl', 'wb'))
model2 = pickle.load(open('anemia_model2.pkl', 'rb'))

pickle.dump(classifier, open('anemia_model.pkl', 'wb'))
model = pickle.load(open('anemia_model.pkl', 'rb'))
