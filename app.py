from flask import Flask, render_template, request
import pandas as pd
from utils import preprocess_input, load_model, generate_report

app = Flask(__name__)
model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_data = {
            "age": float(request.form["age"]),
            "sex": int(request.form["sex"]),
            "cp": int(request.form["cp"]),
            "trestbps": float(request.form["trestbps"]),
            "chol": float(request.form["chol"]),
            "fbs": int(request.form["fbs"]),
            "restecg": int(request.form["restecg"]),
            "thalach": float(request.form["thalach"]),
            "exang": int(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": int(request.form["slope"]),
            "ca": int(request.form["ca"]),
            "thal": int(request.form["thal"])
        }
        df = pd.DataFrame([input_data])
        processed = preprocess_input(df)
        prediction = model.predict(processed)[0]
        generate_report(input_data, prediction)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
