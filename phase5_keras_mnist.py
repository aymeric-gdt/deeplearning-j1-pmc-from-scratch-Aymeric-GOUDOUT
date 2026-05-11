import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Préprocessing : flatten 28x28 → 784, normaliser entre 0 et 1
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test  = X_test.reshape(-1, 784).astype('float32') / 255.0

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Classes uniques : {np.unique(y_train)}")

# TODO : construire le modèle avec keras.Sequential
# architecture : Dense(128, relu) → Dense(64, relu) → Dense(10, softmax)
# préciser input_shape=(784,) sur la première couche
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# TODO : compiler
# optimizer='adam'
# loss='sparse_categorical_crossentropy'  (étiquettes entières, pas one-hot)
# metrics=['accuracy']
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# TODO : entraîner avec model.fit
# epochs=5, batch_size=64, validation_split=0.1, verbose=1
# stocker dans history, mesurer le temps avec time.time()
start = time.time()
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
elapsed = time.time() - start

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTemps d'entraînement : {elapsed:.1f}s")
print(f"Test accuracy        : {test_acc:.4f}")
print(f"Test loss            : {test_loss:.4f}")

# Courbes (code fourni — repose sur history.history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(history.history['accuracy'], label='train')
axes[1].plot(history.history['val_accuracy'], label='val')
axes[1].set_title("Accuracy")
axes[1].legend()
plt.savefig("phase5_mnist_curves.png", dpi=100, bbox_inches='tight')
print("Courbes sauvegardées : phase5_mnist_curves.png")
