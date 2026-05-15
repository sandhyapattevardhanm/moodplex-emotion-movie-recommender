import numpy as np
import pandas as pd
import re
import nltk
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dropout
)

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

nltk.download('punkt')

# ---------------- LOAD DATA ----------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.columns = train_df.columns.str.lower()
test_df.columns = test_df.columns.str.lower()

# ---------------- CLEAN TEXT ----------------
def normalize(text):

    text = str(text)

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = text.lower().strip()

    return text

train_df['text'] = train_df['text'].apply(normalize)
test_df['text'] = test_df['text'].apply(normalize)

# ---------------- LABEL ENCODING ----------------
encoder = LabelEncoder()

train_labels = encoder.fit_transform(train_df['label'])
test_labels = encoder.transform(test_df['label'])

num_classes = len(encoder.classes_)

y_train = to_categorical(train_labels, num_classes=num_classes)
y_test = to_categorical(test_labels, num_classes=num_classes)

# ---------------- TOKENIZER ----------------
VOCAB_SIZE = 10000
MAX_LEN = 50

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(train_df['text'])

train_seq = tokenizer.texts_to_sequences(train_df['text'])
test_seq = tokenizer.texts_to_sequences(test_df['text'])

X_train = pad_sequences(
    train_seq,
    maxlen=MAX_LEN,
    padding='post'
)

X_test = pad_sequences(
    test_seq,
    maxlen=MAX_LEN,
    padding='post'
)

# ---------------- HANDLE IMBALANCE ----------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# ---------------- MODEL ----------------
model = Sequential([

    Embedding(VOCAB_SIZE, 128, input_length=MAX_LEN),

    Conv1D(128, 5, activation='relu'),

    GlobalMaxPooling1D(),

    Dropout(0.5),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- TRAIN ----------------
history = model.fit(

    X_train,
    y_train,

    epochs=15,

    batch_size=32,

    validation_split=0.1,

    class_weight=class_weights
)

# ---------------- EVALUATE ----------------
score = model.evaluate(X_test, y_test)

print("\n✅ TEST ACCURACY:", score[1] * 100)

# ---------------- SAVE ----------------
model.save("emotion_model.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n✅ MODEL SAVED SUCCESSFULLY")