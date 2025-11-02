import xlrd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

filename = 'proyecto_final/extraction/charact_familias.xlsx'
sheetname = 'patterns'

def load_data():
    df = pd.read_excel(filename, sheet_name=sheetname)
    y = df.iloc[:, 0].values.astype(np.float32)
    x = df.iloc[:, 1:].values.astype(np.float32)
    return x, y

accur_list = []
best_accuracy = 0.0
best_model = None

if __name__ == '__main__':
    x, y = load_data()
    print(f'x==y => {len(x)} == {len(y)}: {len(x) == len(y)}')
    if len(x) == len(y):
        print('data loaded correctly !')

        model_sc = StandardScaler()
        model_sc.fit(x)
        x_sc = model_sc.transform(x)

        #! PCA
        model_pca = PCA(n_components=0.98)
        x_pca = model_pca.fit_transform(x_sc)

        # Guardar modelos entrenados
        # joblib.dump(model_sc, 'proyecto_final/train_familias/scaler_model_1_fa.pkl')
        # joblib.dump(model_pca, 'proyecto_final/train_familias/pca_model_1_fa.pkl')

        X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.4, random_state=42, stratify=y)        

        model_mlp = MLPClassifier(
            hidden_layer_sizes=(60, 30),   # 1 hidden layer with 30 neurons
            activation='relu',
            solver='adam',
            alpha=0.001,                # L2 regularization to prevent overfitting
            learning_rate_init=0.001,
            max_iter=6_000,
        )

        scores = cross_validate(model_mlp, x_pca, y, cv=5, return_train_score=True)
        meanCrossVal = np.mean(scores['test_score'])
        print(f'meanCrossVal: {meanCrossVal}')
        
        if meanCrossVal > 0.93: 
            print('data ok')
        
            for i in range(10):
                
                model_mlp.fit(X_train, y_train)
                y_predict = model_mlp.predict(X_test)
                accur = accuracy_score(y_test, y_predict)
                accur_list.append(accur*100)
                print(f'{i+1} : accuracy = {round(accur*100, 2)}%')

                if accur > best_accuracy:
                    best_accuracy = accur
                    best_model = model_mlp
        
        else:   
            print('error in data')
            exit(0)

    else:
        print('error in data')

avg = sum(accur_list)/len(accur_list)   
print('---------------------')     
print(f'avg = {round(avg, 2)}%')

# if best_model is not None:
#     joblib.dump(best_model, 'proyecto_final/train_familias/best_model_1_fa.pkl')
#     print(f'Best model saved with accuracy = {round(best_accuracy*100, 2)}%')