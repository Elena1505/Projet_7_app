from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import pickle

app = Flask(__name__)
data = pd.read_csv("data.csv")
model_path = f"models:/LGBMClassifier/version-4"
print(model_path)
best_model = mlflow.sklearn.load_model(model_uri=model_path)
with open("best_threshold.pickle", "rb") as f:
    thres = pickle.load(f)


def predict(index, data, best_model, thres):
    data_id = data.loc[[index]]
    proba = float(best_model.predict_proba(data_id)[:, 1])
    rep = "Rejetée" if proba > thres else 'Acceptée'
    return rep, round(proba, 2)


@app.route('/', methods=['GET'])
def home():
    return "Home application"


@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        index = request.get_json()
        print(index)
        index = int(index["value"])
        data_index = data.iloc[[index]]
        rep, proba = predict(index, data_index, best_model, thres)
        d = {'rep': rep, 'proba': proba}
        return jsonify(d)
    except Exception as e:
        return jsonify({"erreur": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
