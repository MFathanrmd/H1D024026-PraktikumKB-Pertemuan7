import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

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

def main():
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
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    print("\n[INFO] Menampilkan grafik loss dan accuracy (Tutup window grafik untuk melanjutkan)...")
    pd.DataFrame(history.history).plot(figsize=(10,6))
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.show()

    print("\n[INFO] Melakukan prediksi pada data testing...")
    predictions = model.predict(X_test)
    predicted_classes = predictions.argmax(axis=1)

    print("Prediksi  :", predicted_classes)
    print("Label Asli:", y_test)

    print("\n[INFO] Menampilkan Confusion Matrix (Tutup window grafik untuk melanjutkan)...")
    cm = confusion_matrix(y_test, predicted_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print("\n=== Prediksi Data Baru ===")
    predict_new_data(model, label_encoder)

def predict_new_data(model, label_encoder):
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width = float(input("Masukkan petal width: "))
    
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)
    
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")

if __name__ == '__main__':
    main()