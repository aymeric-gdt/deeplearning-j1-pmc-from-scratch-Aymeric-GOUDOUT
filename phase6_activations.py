import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test  = X_test.reshape(-1, 784).astype('float32') / 255.0

activations = ['sigmoid', 'tanh', 'relu']
results = []
histories = {}

# TODO : pour chaque activation dans activations :
#   1. tf.random.set_seed(42)  (même initialisation pour comparaison équitable)
#   2. construire le modèle [Dense(128, act), Dense(64, act), Dense(10, softmax)]
#   3. compiler : Adam(lr=0.001), sparse_categorical_crossentropy, accuracy
#   4. entraîner 10 epochs, batch=64, validation_split=0.1, verbose=0, mesurer le temps
#   5. évaluer sur X_test
#   6. trouver convergence_epoch = première epoch où val_loss < 0.1 (ou "N/A")
#   7. ajouter un dict à results et stocker val_losses dans histories[activation]
for activation in activations:
    tf.random.set_seed(42)

    model = keras.Sequential([
        keras.layers.Dense(128, activation=activation, input_shape=(784,)),
        keras.layers.Dense(64, activation=activation),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )
    elapsed = time.time() - start

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Trouver epoch de convergence (val_loss < 0.1)
    val_losses = history.history['val_loss']
    convergence_epoch = "N/A"
    for i, vl in enumerate(val_losses):
        if vl < 0.1:
            convergence_epoch = i + 1
            break

    results.append({
        'activation': activation,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'convergence_epoch_sub01': convergence_epoch,
        'train_time_s': elapsed
    })
    histories[activation] = val_losses

# Tableau récapitulatif (code fourni — alimenté par results)
print("\n=== TABLEAU COMPARATIF ===")
print(f"{'Activation':10s} | {'Val loss epoch 10':18s} | {'Test accuracy':14s} | {'Epoch < 0.1 loss':16s} | {'Temps (s)':10s}")
print("-" * 80)
for r in results:
    print(f"{r['activation']:10s} | {r['val_loss_final']:.4f}                 | {r['test_accuracy']:.4f}           | {str(r['convergence_epoch_sub01']):16s} | {r['train_time_s']:.0f}")

# Courbe superposée (code fourni — alimenté par histories)
plt.figure(figsize=(10, 5))
for activation, val_losses in histories.items():
    plt.plot(range(1, 11), val_losses, label=activation, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("Convergence selon la fonction d'activation (MNIST)")
plt.legend()
plt.savefig("phase6_activations_curve.png", dpi=100, bbox_inches='tight')
print("\nCourbe sauvegardée : phase6_activations_curve.png")
