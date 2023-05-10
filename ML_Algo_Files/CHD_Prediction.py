import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("framingham.csv")

data.groupby('TenYearCHD').mean()

X = data.drop(columns='TenYearCHD', axis=1)

data.replace([np.inf, -np.inf], np.nan, inplace=True)

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
Y = data['TenYearCHD']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)

CLF = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = CLF.fit(X_train, X_test, Y_train, Y_test)

models1 = models.sort_values(by=['F1 Score', 'Accuracy'], ascending=False)
models1 = pd.DataFrame(models1)

clf = models1.first_valid_index()
clf1 = GaussianNB()
clf1.fit(X_train, Y_train)

pickle.dump(scaler, open('chd_model2.pkl', 'wb'))
model2 = pickle.load(open('chd_model2.pkl', 'rb'))

pickle.dump(clf1, open('chd_model.pkl', 'wb'))
model = pickle.load(open('chd_model.pkl', 'rb'))
