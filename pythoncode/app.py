from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Flask uygulamasını başlat
app = Flask(__name__)

# Model ve tokenizer'ı yükle
model = load_model("last_saved_model.h5")
with open("last_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Tokenizer'dan gelen metinlerin maksimum uzunluğu
max_len = 400

# Metni sınıflandıran bir API son noktası
@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json  # JSON'dan metni al
    input_text = data.get('text')
    if not input_text:
        return jsonify({"error": "Text is required"}), 400

    # Metni tokenize ve padding işlemleri
    seq = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(seq, maxlen=max_len)

    # Model tahmini
    prediction_prob = model.predict(padded, verbose=0)[0][0]
    label = "normal" if prediction_prob > 0.5 else "fraud"

    return jsonify({
        "text": input_text,
        "prediction": label,
        "confidence": float(prediction_prob)
    })

@app.route('/')
def home():
    return "Flask API is working! Use the /classify endpoint for predictions."


# Uygulamayı çalıştır
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
