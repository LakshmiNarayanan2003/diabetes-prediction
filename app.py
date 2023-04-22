from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis="columns")
y = df[["Outcome"]]
# model = joblib.load("model.joblib")
model = DecisionTreeClassifier()
model.fit(X, y)
# joblib.dump(model, "name.joblib")
app = Flask(__name__)

@app.route("/predict")
def predict():
    q = request.args['q']
    input = [float(x) for x in q.split(",")]
    p = model.predict(np.array([input]))
    prob = model.predict_proba(np.array([input]))
    if p[0] == 1:
        p = 'Patient is diabetic'
    else:
        p = 'Patient is not diabetic'
    return jsonify(
        prediction = str(p),
        accuracy_pos = str(prob[0][0]),
        accuracy_neg = str(prob[0][1])
    )

if __name__ == "__main__":
    app.run(debug=True)

