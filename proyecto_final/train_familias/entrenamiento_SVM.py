import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_excel('proyecto_final/extraction/charact_familias.xlsx', header = None, engine ='openpyxl')

print(df.head())

y = df.iloc[:,0]
x = df.iloc[:,1:]

scaler = StandardScaler()
x_s = scaler.fit_transform(x)

x_train, x_test , y_train, y_test = train_test_split(x_s,y, test_size=0.1, random_state = 42)

model_svm = SVC(kernel = 'rbf' , C = 1.0)

model_svm.fit(x_train,y_train)
y_predict  = model_svm.predict(x_test)
print("accuracy:" , accuracy_score(y_test, y_predict))
print(classification_report(y_test,y_predict, zero_division=0))

#joblib.dump(model_svm,"model_letras_100.pkl")
