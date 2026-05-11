import numpy as np

X = np.array([
    [0.2, 0.1],
    [0.8, 0.9],
    [0.3, 0.7],
    [0.9, 0.2],
])
y = np.array([0, 1, 1, 0])


# TODO : implémenter sigmoid(x)
# formule : 1 / (1 + exp(-x)) avec numpy : np.exp(-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# TODO : implémenter forward(X, w, b)
# étape 1 : somme pondérée  z = X @ w + b  (numpy : np.dot)
# étape 2 : retourner sigmoid(z)
def forward(X, w, b):
    z = X @ w + b
    return sigmoid(z)


# TODO : implémenter compute_loss(y_true, y_pred) — Binary Cross-Entropy
# formule : -mean( y*log(ŷ) + (1-y)*log(1-ŷ) )
# clamper y_pred entre 1e-7 et 1-1e-7 avant le log  → np.clip
def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Poids fixés — pas encore d'entraînement dans cette phase
w = np.array([0.5, -0.3])
b = 0.1

# TODO : appeler forward, puis compute_loss, puis afficher y_pred.round(3) et loss
y_pred = forward(X, w, b)
loss = compute_loss(y, y_pred)

print(f"Prédictions : {y_pred.round(3)}")
print(f"Étiquettes  : {y}")
print(f"Loss BCE    : {loss:.4f}")
