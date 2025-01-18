import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle  # tokenizer ve label encoder'ı kaydetmek için
import re  # Ön işleme için gerekli

# Stopword listesi (İngilizce ve Türkçe sık kullanılan kelimeler)
stopwords = [
    "the", "and", "is", "in", "at", "of", "to", "on", "a", "an", "for", "it", "with",
    "this", "that", "as", "by", "be", "are", "ve", "bir", "bu", "şu", "için", "ile",
    "gibi", "diğer", "da", "de", "ama", "çok", "ile", "her", "ki"
]

def preprocess_text(text):
    # Tüm harfleri küçültmek.
    text = text.lower()
    # Tüm noktalama işaretlerini kaldırmak.
    text = re.sub(r'[^\w\s]', '', text)
    # Belirlediğimiz stopword kelimelerini datasetten kaldırma
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return " ".join(filtered_words)

# 1) Veri Yükleme
data_path = r'C:\Users\bugra\python\FraudDetectedProjectDataSet.xlsx'
data = pd.read_excel(data_path)
data.columns = ['label', 'message']

# Metinleri ön işleme
data['message'] = data['message'].apply(preprocess_text)

X = data['message'].values
y = data['label'].values

# 2) Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 3) Tokenization
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_len)

# 4) Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

# 5) Model Oluşturma
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6) Modeli Eğitme
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=64,
    verbose=1
)

# 7) Modeli Değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# 8) Modeli Kaydetme
model.save("last_saved_model.h5")

# 9) Tokenizer ve LabelEncoder gibi önemli objeleri kaydetme
with open("last_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("last_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model ve ilgili objeler kaydedildi.")
