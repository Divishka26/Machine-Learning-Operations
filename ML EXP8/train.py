import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import joblib

df=pd.read_csv("Iris.csv")
df=df.drop("Id",axis=1)
X=df.drop("Species",axis=1)
y=df['Species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

IrisPredModel=LogisticRegression()
IrisPredModel.fit(X_train,y_train)

pred=IrisPredModel.predict(X_test)

print("Prediction of first sample:",pred[0])

accuracy=accuracy_score(pred,y_test)
print("Accuracy is:",accuracy)

joblib.dump(IrisPredModel,"IrisPredModel.pkl")