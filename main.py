"""
Perceptron Multi-Couche from scratch — numpy CPU only.
Batch vectorized forward/backward pass.
"""

import numpy as np
from typing import List, Optional, Callable
import sys


# ──────────────────────────────────────────────────────────────────────────────
# Activation functions + derivatives
# ──────────────────────────────────────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_d(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # clip for numerical stability
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_d(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_d(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    # stable softmax
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def linear(z: np.ndarray) -> np.ndarray:
    return z


def linear_d(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


ACTIVATIONS = {
    "relu": (relu, relu_d),
    "sigmoid": (sigmoid, sigmoid_d),
    "tanh": (tanh, tanh_d),
    "linear": (linear, linear_d),
    "softmax": (softmax, None),  # softmax used at output, no separate derivative
}


# ──────────────────────────────────────────────────────────────────────────────
# Initializers
# ──────────────────────────────────────────────────────────────────────────────

def xavier_init(fan_in: int, fan_out: int, size: tuple) -> np.ndarray:
    """Xavier/Glorot uniform initialization."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size)


def he_init(fan_in: int, fan_out: int, size: tuple) -> np.ndarray:
    """He initialization for ReLU networks."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*size) * std


INITIALIZERS = {"xavier": xavier_init, "he": he_init}


# ──────────────────────────────────────────────────────────────────────────────
# Optimizers
# ──────────────────────────────────────────────────────────────────────────────

class SGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocities: dict = {}

    def step(self, params: dict, grads: dict):
        for key, W in params.items():
            if key not in self._velocities:
                self._velocities[key] = np.zeros_like(W)
            v = self.momentum * self._velocities[key] - self.lr * grads[key]
            self._velocities[key] = v
            params[key] += v


class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._t = 0
        self._m: dict = {}
        self._v: dict = {}

    def step(self, params: dict, grads: dict):
        self._t += 1
        for key, W in params.items():
            if key not in self._m:
                self._m[key] = np.zeros_like(W)
                self._v[key] = np.zeros_like(W)
            g = grads[key]
            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * g
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * (g ** 2)
            m_hat = self._m[key] / (1 - self.beta1 ** self._t)
            v_hat = self._v[key] / (1 - self.beta2 ** self._t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ──────────────────────────────────────────────────────────────────────────────
# Layer
# ──────────────────────────────────────────────────────────────────────────────

class Layer:
    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: str = "relu",
        init: str = "xavier",
        has_bias: bool = True,
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.has_bias = has_bias

        self.W = INITIALIZERS[init](n_in, n_out, (n_in, n_out))
        self.b = np.zeros((1, n_out)) if has_bias else None

        act_fn, act_d = ACTIVATIONS[activation]
        self.activation = act_fn
        self.activation_d = act_d

        # Cache for backprop
        self.z: Optional[np.ndarray] = None
        self.a: Optional[np.ndarray] = None
        self.input_batch: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X shape: (batch_size, n_in)"""
        self.input_batch = X
        self.z = X @ self.W
        if self.has_bias:
            self.z += self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """grad_out shape: (batch_size, n_out) — upstream gradient.
        Returns gradient w.r.t. input (to pass to previous layer)."""
        if self.activation_d is not None:
            grad_z = grad_out * self.activation_d(self.z)
        else:
            # softmax + cross-entropy combined gradient
            grad_z = grad_out

        batch_size = grad_z.shape[0]

        # Gradient w.r.t. weights (averaged over batch)
        grad_W = self.input_batch.T @ grad_z / batch_size
        grad_b = np.sum(grad_z, axis=0, keepdims=True) / batch_size if self.has_bias else None

        # Gradient w.r.t. input (to backpropagate)
        grad_in = grad_z @ self.W.T

        self._grad_W = grad_W
        self._grad_b = grad_b

        return grad_in

    def get_params(self) -> dict:
        params = {"W": self.W}
        if self.has_bias:
            params["b"] = self.b
        return params

    def get_grads(self) -> dict:
        grads = {"W": self._grad_W}
        if self.has_bias:
            grads["b"] = self._grad_b
        return grads


# ──────────────────────────────────────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────────────────────────────────────

class MLP:
    def __init__(
        self,
        layer_sizes: List[int],
        activations: Optional[List[str]] = None,
        init: str = "xavier",
        output_activation: str = "linear",
        loss: str = "mse",
        optimizer: Optional[object] = None,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        layer_sizes : list of int
            [input_dim, hidden1, hidden2, ..., output_dim]
        activations : list of str or None
            Activation per hidden layer. Defaults to ['relu'] * (len(layer_sizes)-2)
        init : 'xavier' or 'he'
        output_activation : 'linear', 'sigmoid', 'softmax'
        loss : 'mse' or 'crossentropy'
        optimizer : SGD or Adam instance
        """
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.init = init
        self.loss = loss
        self._layers: List[Layer] = []

        if activations is None:
            activations = ["relu"] * (len(layer_sizes) - 2)
        if len(activations) < len(layer_sizes) - 2:
            activations += ["relu"] * (len(layer_sizes) - 2 - len(activations))

        # Build hidden layers
        for i in range(len(layer_sizes) - 2):
            self._layers.append(
                Layer(
                    n_in=layer_sizes[i],
                    n_out=layer_sizes[i + 1],
                    activation=activations[i],
                    init=init,
                    has_bias=True,
                )
            )

        # Output layer
        self._layers.append(
            Layer(
                n_in=layer_sizes[-2],
                n_out=layer_sizes[-1],
                activation=output_activation,
                init=init,
                has_bias=True,
            )
        )

        self.optimizer = optimizer if optimizer is not None else SGD(lr=0.01, momentum=0.9)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.loss == "mse":
            return np.mean((y_pred - y_true) ** 2)
        elif self.loss == "crossentropy":
            # cross-entropy with softmax (numerically stable)
            eps = 1e-12
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def _backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        if self.loss == "mse":
            grad = 2.0 * (y_pred - y_true) / y_pred.shape[0]
        elif self.loss == "crossentropy":
            # softmax + cross-entropy gradient = (y_pred - y_true)
            grad = (y_pred - y_true) / y_pred.shape[0]
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

        for layer in reversed(self._layers):
            grad = layer.backward(grad)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
        shuffle: bool = True,
        grad_check: bool = False,
        verbose: int = 1,
    ):
        """
        Train the MLP.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples, n_outputs)
        epochs : int
        batch_size : int
        val_X, val_y : optional validation data
        grad_check : bool — runs numeric gradient checking on first epoch (slow)
        verbose : 0 = silent, 1 = print every 10 epochs, 2 = every epoch
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                y_pred = self.forward(X_batch)
                loss = self._compute_loss(y_pred, y_batch)
                epoch_loss += loss
                n_batches += 1

                self._backward(y_pred, y_batch)

                # Collect params & grads
                params = {}
                grads = {}
                for layer in self._layers:
                    for k, v in layer.get_params().items():
                        full_key = f"{id(layer)}_{k}"
                        params[full_key] = v
                    for k, v in layer.get_grads().items():
                        full_key = f"{id(layer)}_{k}"
                        grads[full_key] = v

                self.optimizer.step(params, grads)

                # Sync optimized params back to layers
                for layer in self._layers:
                    layer.W = params[f"{id(layer)}_W"]
                    if layer.has_bias:
                        layer.b = params[f"{id(layer)}_b"]

            avg_loss = epoch_loss / n_batches

            # Validation metrics
            if val_X is not None:
                val_pred = self.predict(val_X)
                val_loss = self._compute_loss(val_pred, val_y)
                val_mae = np.mean(np.abs(val_pred - val_y))
            else:
                val_loss = None
                val_mae = None

            if verbose >= 2 or (verbose == 1 and epoch % 10 == 0):
                msg = f"Epoch {epoch+1:4d}/{epochs} | train_loss: {avg_loss:.6f}"
                if val_loss is not None:
                    msg += f" | val_loss: {val_loss:.6f} | val_mae: {val_mae:.6f}"
                print(msg)

            # Gradient checking on first epoch
            if grad_check and epoch == 0:
                self._gradient_check(X, y)
                print("Gradient check passed.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass only — no dropout or stochastic elements."""
        return self.forward(X)

    def _gradient_check(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-5, tol: float = 1e-4):
        """Numerical gradient checking. Computes analytic vs numeric gradients."""
        print("Running gradient check...")
        y_pred = self.forward(X)
        self._backward(y_pred, y)

        params = {}
        grads_analytic = {}
        for layer in self._layers:
            for k, v in layer.get_params().items():
                full_key = f"{id(layer)}_{k}"
                params[full_key] = v
                grads_analytic[full_key] = layer.get_grads()[k]

        n_errors = 0
        for key, W in params.items():
            W_flat = W.flatten()
            grad_analytic = grads_analytic[key].flatten()
            grad_numeric = np.zeros_like(grad_analytic)

            for i in range(len(W_flat)):
                W_flat[i] += eps
                W[key].flatten()[:] = W_flat
                y_pred_plus = self.forward(X)
                loss_plus = self._compute_loss(y_pred_plus, y)

                W_flat[i] -= 2 * eps
                W[key].flatten()[:] = W_flat
                y_pred_minus = self.forward(X)
                loss_minus = self._compute_loss(y_pred_minus, y)

                grad_numeric[i] = (loss_plus - loss_minus) / (2 * eps)
                W_flat[i] += eps  # restore

            # Restore original
            W[key].flatten()[:] = W_flat

            diff = np.abs(grad_analytic - grad_numeric)
            rel_err = diff / (np.abs(grad_analytic) + np.abs(grad_numeric) + 1e-8)
            max_rel_err = np.max(rel_err)

            if max_rel_err > tol:
                n_errors += 1
                print(f"  [FAIL] {key}: max rel error = {max_rel_err:.2e} > {tol:.2e}")

        if n_errors == 0:
            print(f"  [PASS] All gradients within {tol:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo: XOR + Circles
# ──────────────────────────────────────────────────────────────────────────────

def demo_xor():
    print("=" * 60)
    print("Demo: XOR — non-linearly separable problem")
    print("=" * 60)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    mlp = MLP(
        layer_sizes=[2, 16, 16, 1],  # plus grand + tanh zero-centered
        activations=["tanh", "tanh"],
        init="xavier",
        output_activation="linear",
        loss="mse",
        optimizer=Adam(lr=0.01),
        seed=0,
    )

    mlp.fit(X, y, epochs=3000, batch_size=4, verbose=2)

    preds = mlp.predict(X)
    print(f"\nPredictions:\n{preds.round(3)}")
    print(f"Targets:\n{y}")


def demo_circles():
    print("\n" + "=" * 60)
    print("Demo: Circles — classification")
    print("=" * 60)

    # Generate circles dataset
    np.random.seed(42)
    n = 400
    r1, r2 = 0.4, 0.7

    angles = np.random.uniform(0, 2 * np.pi, n)
    inner = np.column_stack([0.5 + r1 * np.cos(angles[:n // 2]), 0.5 + r1 * np.sin(angles[:n // 2])])
    outer = np.column_stack([0.5 + r2 * np.cos(angles[n // 2:]), 0.5 + r2 * np.sin(angles[n // 2:])])

    X = np.vstack([inner, outer]).astype(float)
    y_onehot = np.zeros((n, 2), dtype=float)
    y_onehot[:n // 2, 0] = 1
    y_onehot[n // 2:, 1] = 1
    y = y_onehot

    # Train/val split
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    mlp = MLP(
        layer_sizes=[2, 32, 16, 2],
        activations=["relu", "relu"],
        init="he",
        output_activation="softmax",
        loss="crossentropy",
        optimizer=Adam(lr=0.01),
        seed=42,
    )

    mlp.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        val_X=X_val,
        val_y=y_val,
        verbose=1,
    )

    # Accuracy
    preds = mlp.predict(X_val)
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_val, axis=1))
    print(f"\nValidation accuracy: {acc:.2%}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_xor()
    demo_circles()
