import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load the dataset
try:
    df = pd.read_csv('Emotion_classify_Data.csv')
except FileNotFoundError:
    print("Error: 'Emotion_classify_Data.csv' not found.\nPlease download the 'Emotion Dataset' from Kaggle user 'ABDALLAH WAGIH IBRAHIM'\nand extract it to the root directory of this project.")
    import sys
    sys.exit()
comments = df['Comment'].values
emotions = pd.get_dummies(df['Emotion']).values

# Preprocess the text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
padded_sequences = pad_sequences(sequences, maxlen=200)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, emotions, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=16),
    LSTM(64),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=11, validation_data=(X_test, y_test)) # Changed epochs from 10 to 11

# Save the model
model.save('trained_model/emotion_classifier.h5')
