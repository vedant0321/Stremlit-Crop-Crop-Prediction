import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Crop_recommendation.csv")


label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])


X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values
y = data['label_encoded'].values


y = tf.keras.utils.to_categorical(y, num_classes=22)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(22, activation='softmax')  # 22 output neurons for 22 crops
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


model.save("crop_recommendation_model.h5")
