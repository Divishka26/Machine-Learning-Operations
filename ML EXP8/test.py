import joblib
from sklearn.metrics import accuracy_score
import pandas as pd
model = joblib.load("IrisPredModel.pkl")
test_df = pd.read_csv("Iris.csv")
X = test_df.drop(["Id", "Species"], axis=1)
y = test_df['Species']
pred = model.predict(X)
acc = accuracy_score(pred, y)
print(acc)

assert len(pred)==len(y)
assert acc>=0.50
print("all tests passed")