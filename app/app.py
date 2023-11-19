import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disables GPU usage

from flask import Flask, request, render_template
import unicodedata, re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
model = load_model('app/sentiment_analysis_model.h5')
with open('app/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract text from the form
        text = request.form['text']

        # Get the prediction
        prediction = predict_sentiment(text)

        # Render the template with the prediction result
        return render_template('result.html', prediction=prediction)

    # For a GET request, just render the form
    return render_template('index.html')

    
def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?!,¿]+", " ", w)
    w = w.strip()
    w = "<start> " + w + " <end>"
    return w

def predict_sentiment(text):
    # Preprocess the input text
    processed_text = preprocess_sentence(text)

    # Tokenize and pad the processed text
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding="post")

    # Make a prediction
    prediction = model.predict(padded_sequences)
    prediction_label = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return prediction_label

if __name__ == '__main__':
    app.run(debug=False)