from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Training
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train, y_train)


# Web page
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Email Spam Classifier</title>

<style>
body {
    font-family: Arial;
    background: #eef2ff;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}

.box {
    background: white;
    padding: 30px;
    width: 500px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 10px 30px #aaa;
}

textarea {
    width: 95%;
    height: 150px;
    padding: 12px;
    border-radius: 10px;
}

button {
    width: 100%;
    padding: 13px;
    margin-top: 12px;
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 16px;
}

.result {
    margin-top: 20px;
    padding: 15px;
    border-radius: 10px;
    background: #eee;
}
</style>
</head>

<body>

<div class="box">

<h1>📧 Email Spam Classifier</h1>

<p>Machine Learning Based Spam Detection</p>

<form method="POST">

<textarea name="message"
placeholder="Enter email message..."
required></textarea>

<br>

<button type="submit">🔍 Check Email</button>

</form>

{% if result %}

<div class="result">

<h2>{{ result }}</h2>

<p>Confidence: {{ confidence }}%</p>

</div>

{% endif %}

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None

    if request.method == "POST":

        message = request.form["message"]

        data = vectorizer.transform([message])

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0]

        confidence = round(max(probability) * 100, 2)

        if prediction == "spam":
            result = "⚠️ SPAM EMAIL"
        else:
            result = "✅ HAM EMAIL"

    return render_template_string(
        HTML,
        result=result,
        confidence=confidence
    )


app.run(debug=True)