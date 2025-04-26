# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries and load dataset.
2. Split dataset into Training and Testing Sets.
3. Train the model using Stochastic Gradient Decent(SGD).
4. Make Predictions and Evaluate Accuracy.
5. Generate confusion matrix.

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Vijayaraj V
RegisterNumber:  212222230174
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

iris=load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

print(df.tail())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```

## Output:
![image](https://github.com/user-attachments/assets/79dfff62-2844-4f02-9a59-2dd4ac76d8a7)

![image](https://github.com/user-attachments/assets/1fe1ca1a-7e2c-4192-b404-ef56c13aaf85)

![image](https://github.com/user-attachments/assets/4786aaff-c4dc-4c02-9241-ab8d714dc48b)

![image](https://github.com/user-attachments/assets/e4aa2057-9a0b-4c34-a045-811ceb1ebf2a)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
