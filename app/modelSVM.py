import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('heart.csv')

X = data.drop(['target', 'age', 'exang'],axis=1) #dropping target out of data
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

svc_model = SVC()

X_train_min = X_train.min()
X_train_max = X_train.max()
X_train_range = X_train_max - X_train_min
X_train_scaled = (X_train - X_train_min) / (X_train_range)

X_test_min = X_test.min()
X_test_max = X_test.max()
X_test_range = (X_test_max - X_test_min)
X_test_scaled = (X_test - X_test_min) / X_test_range

svc_model.fit(X_train_scaled, y_train)

y_predict = svc_model.predict(X_test_scaled)
print('{}'.format(X_test_scaled))
conf_matrix = confusion_matrix(y_test, y_predict)
conf_matrix = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
conf_matrix_df = pd.DataFrame(conf_matrix, index=['heart_disease','healthy'], columns=['predicted_heart_disease', 'predicted_healthy'])
conf_matrix_df

print(classification_report(y_test, y_predict))

pickle.dump(svc_model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
