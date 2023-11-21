import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import psutil
from flask import Flask, request, render_template
import unicodedata, re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

tf.get_logger().setLevel('ERROR') 
app = Flask(__name__)

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

def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    
print("Memory usage BEFORE loading model and tokenizer")    
log_memory_usage()

model = load_model('app/sentiment_analysis_model.h5')
with open('app/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
print("Memory usage AFTER loading model and tokenizer")    
log_memory_usage()

def predict_sentiment(text):
    print("Memory usage BEFORE prediction")  
    log_memory_usage()
    # Preprocess the input text
    processed_text = preprocess_sentence(text)

    # Tokenize and pad the processed text
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding="post")

    # Make a prediction
    prediction = model.predict(padded_sequences)
    prediction_label = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    print("Memory usage AFTER prediction")
    log_memory_usage()  
    return prediction_label

if __name__ == '__main__':
    app.run(debug=False)