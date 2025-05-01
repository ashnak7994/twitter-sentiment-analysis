from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load model and tokenizer
model = load_model("lstm_sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
emoji_map = {'Negative': '😞', 'Neutral': '😐', 'Positive': '😊'}

MAX_LEN = 100

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    emoji = None

    if request.method == "POST":
        text = request.form["tweet"].strip().lower()

        if not text:
            prediction = "Please enter a tweet."
            emoji = "⚠️"
        else:
            seq = tokenizer.texts_to_sequences([text])
            padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
            pred = model.predict(padded)
            label_index = np.argmax(pred)
            label = label_map[label_index]
            prediction = label
            emoji = emoji_map[label]

    return render_template("index.html", prediction=prediction, emoji=emoji)

if __name__ == "__main__":
    app.run(debug=True)
