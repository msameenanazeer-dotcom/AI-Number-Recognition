from flask import Flask, render_template, request
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

digits = load_digits()

model = KNeighborsClassifier(n_neighbors=3)
model.fit(digits.data, digits.target)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        values = request.form["pixels"].split(",")

        pixels = np.array([float(x) for x in values])

        if len(pixels) == 64:
            prediction = model.predict([pixels])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
