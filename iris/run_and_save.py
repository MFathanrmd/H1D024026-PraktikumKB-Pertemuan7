import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("=== Praktikum KB Pertemuan 7: Jaringan Syaraf Tiruan 2 ===")

print("\n[INFO] Memuat dataset Iris dari file lokal...")
dataset = pd.read_csv('iris/iris.data', header=None, sep=',')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("[INFO] Mengonversi label...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("[INFO] Membagi dataset (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

print("\n[INFO] Membangun Arsitektur Model JST...")
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])
model.summary()

print("\n[INFO] Mengkompilasi model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n[INFO] Memulai pelatihan model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("\n[INFO] Mengevaluasi model pada data testing...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Save training history plot
print("\n[INFO] Menyimpan grafik training history...")
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.tight_layout()
plt.savefig('iris/training_history.png', dpi=150)
plt.close()
print("  Grafik disimpan: iris/training_history.png")

print("\n[INFO] Melakukan prediksi pada data testing...")
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

print("Prediksi  :", predicted_classes)
print("Label Asli:", y_test)

# Accuracy per sample
correct = (predicted_classes == y_test).sum()
print(f"\nPrediksi benar: {correct}/{len(y_test)} ({correct/len(y_test)*100:.1f}%)")

# Save confusion matrix
print("\n[INFO] Menyimpan Confusion Matrix...")
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('iris/confusion_matrix.png', dpi=150)
plt.close()
print("  Confusion matrix disimpan: iris/confusion_matrix.png")

print("\n=== SELESAI ===")
print(f"Akurasi akhir: {accuracy*100:.2f}%")
print(f"Loss akhir   : {loss:.4f}")
