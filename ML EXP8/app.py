import joblib
from flask import Flask, request, jsonify

app=Flask(__name__)

model=joblib.load("IrisPredModel.pkl")

@app.route("/")
def home():
    return jsonify({"message":"App is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data=request.get_json()
    s_l=data["sepal length"]
    s_w=data["sepal width"]
    p_l=data["petal length"]
    p_w=data["petal width"]
    pred=model.predict(([s_l,s_w,p_l,p_w]))
    return jsonify({"prediction":pred[0]})

# _name__ is a special variable of python file created by python interpreter itself in background, its value is always main when this file is called/executed
if __name__=="__main__":
 app.run(host="0.0.0.0", port=5001)