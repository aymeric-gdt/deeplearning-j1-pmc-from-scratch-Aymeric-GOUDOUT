# PMC Batched — Perceptron Multi-Couche from scratch

MLP pur numpy, calcul vectorisé par batch — pas de autograd, pas de framework.

## Architecture

```
Input (n_features)
    ↓
Layer 1: n_in → n_out, activation, bias, Xavier/He init
    ↓
Layer 2: ...
    ↓
Output Layer: activation output, bias
    ↓
Loss (MSE ou CrossEntropy)
```

## Calcul batch matriciel

Tout le forward et backward est vectorisé en matrices 2D — aucune boucle sur les exemples.

**Forward pass** d'une couche :
```
Z = X @ W + b        # (batch_size, n_in) @ (n_in, n_out) + (1, n_out)
A = activation(Z)   # element-wise
```
Une seule multiplication matricielle couvre l'intégralité du batch.

**Backward pass** :
```
grad_Z = grad_out * activation'(Z)   # element-wise
grad_W = X.T @ grad_Z / batch_size   # (n_in, batch) @ (batch, n_out)
grad_b = sum(grad_Z, axis=0)
grad_in = grad_Z @ W.T               # pour la couche précédente
```

Pas de boucle `for sample in batch` — tout est une opération matricielle.

## Fonctionnalités

- **Activations** : ReLU, Sigmoid, Tanh, Linear, Softmax
- **Loss** : MSE, CrossEntropy (softmax intégré)
- **Optimizers** : SGD, SGD + Momentum, Adam
- **Initializers** : Xavier, He
- **Gradient checking** numérique (optionnel, epoch 1)
- **Bias** sur chaque couche
- **Validation split** intégré dans `fit()`

## API

```python
from main import MLP, Adam

mlp = MLP(
    layer_sizes=[2, 32, 16, 2],
    activations=["relu", "relu"],
    output_activation="softmax",
    loss="crossentropy",
    optimizer=Adam(lr=0.01),
    seed=42,
)

mlp.fit(X_train, y_train,
        epochs=200,
        batch_size=32,
        val_X=X_val,
        val_y=y_val,
        verbose=1)

preds = mlp.predict(X_test)
```

## Demos

- `demo_xor()` — XOR (non-linéaire)
- `demo_circles()` — classification anneaux concentriques

```bash
python main.py
```

## Dépendances

```
numpy>=1.24.0
```
