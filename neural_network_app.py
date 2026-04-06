import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_fn(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z) ** 2

ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh_fn, tanh_deriv),
}


def init_network(layer_sizes):
    params = {}
    for i in range(1, len(layer_sizes)):
        scale = np.sqrt(2.0 / layer_sizes[i - 1])
        params[f"W{i}"] = np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * scale
        params[f"b{i}"] = np.zeros((1, layer_sizes[i]))
    return params


def forward(X_data, params, n_hidden, act_fn):
    cache = {"A0": X_data}
    A = X_data
    for i in range(1, n_hidden + 1):
        Z = A @ params[f"W{i}"] + params[f"b{i}"]
        A = act_fn(Z)
        cache[f"Z{i}"] = Z
        cache[f"A{i}"] = A
    Z_out = A @ params[f"W{n_hidden + 1}"] + params[f"b{n_hidden + 1}"]
    A_out = sigmoid(Z_out)
    cache[f"Z{n_hidden + 1}"] = Z_out
    cache[f"A{n_hidden + 1}"] = A_out
    return A_out, cache


def compute_loss(Y_pred, Y_true, method="cross_entropy"):
    eps = 1e-8
    Y_pred = np.clip(Y_pred, eps, 1 - eps)
    if method == "cross_entropy":
        return -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
    else:
        return np.mean((Y_pred - Y_true) ** 2)


def backward(Y_true, params, cache, n_hidden, act_deriv, loss_method="cross_entropy"):
    grads = {}
    m = Y_true.shape[0]
    L = n_hidden + 1
    A_out = cache[f"A{L}"]

    if loss_method == "cross_entropy":
        dZ = A_out - Y_true
    else:
        dZ = 2 * (A_out - Y_true) * sigmoid_deriv(cache[f"Z{L}"])

    grads[f"dW{L}"] = (cache[f"A{L - 1}"].T @ dZ) / m
    grads[f"db{L}"] = np.mean(dZ, axis=0, keepdims=True)
    dA = dZ @ params[f"W{L}"].T

    for i in range(L - 1, 0, -1):
        dZ = dA * act_deriv(cache[f"Z{i}"])
        grads[f"dW{i}"] = (cache[f"A{i - 1}"].T @ dZ) / m
        grads[f"db{i}"] = np.mean(dZ, axis=0, keepdims=True)
        if i > 1:
            dA = dZ @ params[f"W{i}"].T

    return grads


def update_params(params, grads, lr, n_hidden):
    for i in range(1, n_hidden + 2):
        params[f"W{i}"] = params[f"W{i}"] - lr * grads[f"dW{i}"]
        params[f"b{i}"] = params[f"b{i}"] - lr * grads[f"db{i}"]
    return params


def train_network(X_data, Y_data, layer_sizes, lr, epochs, act_name, loss_method):
    act_fn, act_deriv = ACTIVATIONS[act_name]
    n_hidden = len(layer_sizes) - 2
    np.random.seed(42)
    params = init_network(layer_sizes)
    history = {"loss": [], "accuracy": [], "params_snapshots": [], "grads_snapshots": []}

    for epoch in range(epochs):
        Y_pred, cache = forward(X_data, params, n_hidden, act_fn)
        loss = compute_loss(Y_pred, Y_data, loss_method)
        grads = backward(Y_data, params, cache, n_hidden, act_deriv, loss_method)
        params = update_params(params, grads, lr, n_hidden)

        preds = (Y_pred >= 0.5).astype(int)
        acc = np.mean(preds == Y_data)

        history["loss"].append(loss)
        history["accuracy"].append(acc)

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            snapshot = {k: v.copy() for k, v in params.items()}
            history["params_snapshots"].append((epoch, snapshot))
            grad_snapshot = {k: v.copy() for k, v in grads.items()}
            history["grads_snapshots"].append((epoch, grad_snapshot))

    return params, history, cache


def predict(X_data, params, n_hidden, act_fn):
    A = X_data
    for i in range(1, n_hidden + 1):
        Z = A @ params[f"W{i}"] + params[f"b{i}"]
        A = act_fn(Z)
    Z_out = A @ params[f"W{n_hidden + 1}"] + params[f"b{n_hidden + 1}"]
    return sigmoid(Z_out)


def plot_decision_boundary_nn(params, X_data, y_data, n_hidden, act_fn, title, ax):
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(grid, params, n_hidden, act_fn)
    Z = (Z >= 0.5).astype(int).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap=plt.cm.RdYlBu, edgecolors='k', s=20)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def run():
    st.title("Neural Network Explorer")

    st.sidebar.header("Neural Network Controls")

    dataset_name = st.sidebar.selectbox("Dataset", ["Linear", "Moons (Non-linear)", "Circles", "XOR-like", "Noisy Moons"], key="nn_dataset")
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 2.0, 0.5, step=0.01, key="nn_lr")
    n_layers = st.sidebar.slider("Hidden Layers", 1, 4, 2, key="nn_layers")
    neurons_per_layer = st.sidebar.slider("Neurons per Layer", 2, 32, 8, key="nn_neurons")
    n_epochs = st.sidebar.slider("Epochs", 10, 2000, 500, step=10, key="nn_epochs")
    activation_name = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"], key="nn_act")
    loss_name = st.sidebar.selectbox("Loss Function", ["cross_entropy", "mse"], key="nn_loss")
    add_noise = st.sidebar.toggle("Add Noise", False, key="nn_noise")
    show_gradients = st.sidebar.toggle("Show Gradients", False, key="nn_grads")

    np.random.seed(42)
    n_samples = 300

    if dataset_name == "Linear":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
    elif dataset_name == "Moons (Non-linear)":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif dataset_name == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=0.15, factor=0.5, random_state=42)
    elif dataset_name == "XOR-like":
        X = np.random.randn(n_samples, 2) * 1.5
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    else:
        X, y = make_moons(n_samples=n_samples, noise=0.4, random_state=42)

    if add_noise:
        X = X + np.random.randn(*X.shape) * 0.3

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Y_train = y_train.reshape(-1, 1)
    Y_test = y_test.reshape(-1, 1)

    layer_sizes = [2] + [neurons_per_layer] * n_layers + [1]
    act_fn, act_deriv = ACTIVATIONS[activation_name]

    params, history, cache = train_network(X_train, Y_train, layer_sizes, learning_rate, n_epochs, activation_name, loss_name)

    train_pred = (predict(X_train, params, n_layers, act_fn) >= 0.5).astype(int)
    test_pred = (predict(X_test, params, n_layers, act_fn) >= 0.5).astype(int)
    final_train_acc = np.mean(train_pred == Y_train)
    final_test_acc = np.mean(test_pred == Y_test)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Neuron Computation", "Activation Functions", "Network Architecture",
        "Forward Propagation", "Loss Visualization", "Backpropagation", "Model Complexity"
    ])

    with tab1:
        st.subheader("Neuron Computation: Weighted Sum + Activation")
        st.write("A single neuron takes inputs, multiplies each by a weight, adds a bias, then passes the result through an activation function. This is the fundamental building block.")
        st.info("**What to observe:** Adjust the weights and bias sliders to see how the neuron's output changes. The weighted sum is a linear combination, and the activation adds non-linearity.")

        col1, col2 = st.columns(2)
        with col1:
            w1 = st.slider("Weight 1", -3.0, 3.0, 1.0, step=0.1, key="nn_w1")
            w2 = st.slider("Weight 2", -3.0, 3.0, -0.5, step=0.1, key="nn_w2")
            bias = st.slider("Bias", -3.0, 3.0, 0.0, step=0.1, key="nn_bias")

        x1_val = 0.8
        x2_val = -0.3
        z_val = w1 * x1_val + w2 * x2_val + bias
        a_val = ACTIVATIONS[activation_name][0](np.array([z_val]))[0]

        with col2:
            st.write(f"**Input x1:** {x1_val}")
            st.write(f"**Input x2:** {x2_val}")
            st.write(f"**Weighted sum z:** ({w1} x {x1_val}) + ({w2} x {x2_val}) + ({bias}) = **{z_val:.4f}**")
            st.write(f"**Activation ({activation_name}):** a = {activation_name}({z_val:.4f}) = **{a_val:.4f}**")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        z_range = np.linspace(-5, 5, 200)
        a_range = ACTIVATIONS[activation_name][0](z_range)
        axes[0].plot(z_range, a_range, 'b-', linewidth=2)
        axes[0].axvline(x=z_val, color='red', linestyle='--', linewidth=1.5, label=f'z = {z_val:.2f}')
        axes[0].axhline(y=a_val, color='green', linestyle='--', linewidth=1.5, label=f'a = {a_val:.2f}')
        axes[0].scatter([z_val], [a_val], c='red', s=100, zorder=5)
        axes[0].set_xlabel("z (weighted sum)")
        axes[0].set_ylabel("a (activation output)")
        axes[0].set_title(f"Single Neuron Output ({activation_name})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        x1_range = np.linspace(-3, 3, 100)
        x2_range = np.linspace(-3, 3, 100)
        xx1, xx2 = np.meshgrid(x1_range, x2_range)
        zz = w1 * xx1 + w2 * xx2 + bias
        aa = ACTIVATIONS[activation_name][0](zz)
        contour = axes[1].contourf(xx1, xx2, aa, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=axes[1], label='Activation Output')
        axes[1].scatter([x1_val], [x2_val], c='red', s=100, marker='*', zorder=5)
        axes[1].set_xlabel("x1")
        axes[1].set_ylabel("x2")
        axes[1].set_title("Neuron Output Across Input Space")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Activation Functions Compared")
        st.write("Activation functions decide what signal a neuron passes forward. Without them, the whole network would just be a single linear function no matter how many layers you stack.")
        st.info("**What to observe:** ReLU kills negative inputs (sets to 0). Sigmoid squashes everything to [0,1]. Tanh squashes to [-1,1]. Notice where each one saturates (flat regions = vanishing gradients).")

        z_range = np.linspace(-5, 5, 300)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for idx, (name, (fn, deriv)) in enumerate(ACTIVATIONS.items()):
            axes[0][idx].plot(z_range, fn(z_range), 'b-', linewidth=2.5)
            axes[0][idx].set_title(f"{name.upper()}", fontsize=14)
            axes[0][idx].set_xlabel("z")
            axes[0][idx].set_ylabel("activation(z)")
            axes[0][idx].grid(True, alpha=0.3)
            axes[0][idx].axhline(y=0, color='k', linewidth=0.5)
            axes[0][idx].axvline(x=0, color='k', linewidth=0.5)

            axes[1][idx].plot(z_range, deriv(z_range), 'r-', linewidth=2.5)
            axes[1][idx].set_title(f"{name.upper()} Derivative", fontsize=14)
            axes[1][idx].set_xlabel("z")
            axes[1][idx].set_ylabel("derivative")
            axes[1][idx].grid(True, alpha=0.3)
            axes[1][idx].axhline(y=0, color='k', linewidth=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.write("**Quick comparison:**")
        comparison_data = [
            {"Function": "ReLU", "Range": "[0, inf)", "Zero-centered": "No", "Vanishing Gradient": "No (for z > 0)", "Common Use": "Hidden layers (default choice)"},
            {"Function": "Sigmoid", "Range": "(0, 1)", "Zero-centered": "No", "Vanishing Gradient": "Yes (at extremes)", "Common Use": "Output layer (binary classification)"},
            {"Function": "Tanh", "Range": "(-1, 1)", "Zero-centered": "Yes", "Vanishing Gradient": "Yes (at extremes)", "Common Use": "Hidden layers (alternative to ReLU)"},
        ]
        st.dataframe(comparison_data, use_container_width=True)

    with tab3:
        st.subheader("Network Architecture")
        st.write("Data flows from the input layer (your features) through hidden layers (where patterns are learned) to the output layer (the prediction). Each connection has a weight.")
        st.info("**What to observe:** More layers and neurons increase capacity but also risk overfitting. The network diagram shows the current architecture you've configured in the sidebar.")

        fig = go.Figure()

        all_layers = [2] + [neurons_per_layer] * n_layers + [1]
        layer_names = ["Input"] + [f"Hidden {i + 1}" for i in range(n_layers)] + ["Output"]
        max_neurons = max(all_layers)
        n_total_layers = len(all_layers)

        node_positions = {}
        for l_idx, n_neurons in enumerate(all_layers):
            x = l_idx / (n_total_layers - 1) if n_total_layers > 1 else 0.5
            for n_idx in range(n_neurons):
                y_pos = (n_idx + 0.5) / max_neurons if max_neurons > 0 else 0.5
                if n_neurons < max_neurons:
                    offset = (max_neurons - n_neurons) / (2 * max_neurons)
                    y_pos = offset + n_idx / max_neurons
                node_positions[(l_idx, n_idx)] = (x, y_pos)

        edge_x, edge_y = [], []
        for l_idx in range(len(all_layers) - 1):
            for n1 in range(all_layers[l_idx]):
                for n2 in range(all_layers[l_idx + 1]):
                    x1, y1 = node_positions[(l_idx, n1)]
                    x2, y2 = node_positions[(l_idx + 1, n2)]
                    edge_x += [x1, x2, None]
                    edge_y += [y1, y2, None]

        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='rgba(150,150,150,0.3)', width=0.5), hoverinfo='none'))

        for l_idx, n_neurons in enumerate(all_layers):
            nx = [node_positions[(l_idx, n)][0] for n in range(n_neurons)]
            ny = [node_positions[(l_idx, n)][1] for n in range(n_neurons)]

            if l_idx == 0:
                color = '#3498db'
            elif l_idx == len(all_layers) - 1:
                color = '#e74c3c'
            else:
                color = '#2ecc71'

            hover_text = [f"{layer_names[l_idx]}\nNeuron {n + 1}" for n in range(n_neurons)]
            fig.add_trace(go.Scatter(x=nx, y=ny, mode='markers',
                                     marker=dict(size=20, color=color, line=dict(width=2, color='black')),
                                     hovertext=hover_text, hoverinfo='text', name=layer_names[l_idx]))

        fig.update_layout(showlegend=True, height=500,
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(len(all_layers))
        for i, (lname, size) in enumerate(zip(layer_names, all_layers)):
            cols[i].metric(lname, f"{size} neurons")

        total_params = sum(all_layers[i] * all_layers[i + 1] + all_layers[i + 1] for i in range(len(all_layers) - 1))
        st.write(f"**Total trainable parameters:** {total_params}")

    with tab4:
        st.subheader("Forward Propagation")
        st.write("In forward propagation, data enters the input layer and flows through each hidden layer. At each layer, we compute weighted sums and apply activations. The final output is the prediction.")
        st.info("**What to observe:** See how the activations at each layer transform the data. The decision boundary shows the result of the full forward pass on a grid of points.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='k', s=20)
        axes[0].set_title("Input Data (Before Network)")
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")

        plot_decision_boundary_nn(params, X_train, y_train, n_layers, act_fn, f"After Training ({n_epochs} epochs)", axes[1])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.write("**Layer-by-layer activations (sample of training data):**")

        sample_X = X_train[:5]
        A = sample_X
        fig2, axes2 = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))
        if n_layers + 1 == 1:
            axes2 = [axes2]

        for i in range(1, n_layers + 1):
            Z = A @ params[f"W{i}"] + params[f"b{i}"]
            A = act_fn(Z)
            if i - 1 < len(axes2):
                axes2[i - 1].imshow(A.T, aspect='auto', cmap='viridis')
                axes2[i - 1].set_title(f"Hidden Layer {i} Activations")
                axes2[i - 1].set_xlabel("Sample")
                axes2[i - 1].set_ylabel("Neuron")

        Z_out = A @ params[f"W{n_layers + 1}"] + params[f"b{n_layers + 1}"]
        A_out = sigmoid(Z_out)
        axes2[-1].bar(range(len(sample_X)), A_out.flatten(), color=['#e74c3c' if v < 0.5 else '#3498db' for v in A_out.flatten()])
        axes2[-1].set_title("Output Probabilities")
        axes2[-1].set_xlabel("Sample")
        axes2[-1].set_ylabel("P(class=1)")
        axes2[-1].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with tab5:
        st.subheader("Loss Visualization")
        st.write("The loss function measures how wrong the predictions are. During training, we want this number to go down. Cross-entropy penalizes confident wrong predictions heavily, while MSE is smoother.")
        st.info("**What to observe:** Watch the loss curve drop. A smooth, steady decrease means good learning. Spikes or plateaus may indicate the learning rate is too high or too low.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(history["loss"], 'b-', linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"Training Loss ({loss_name})")
        axes[0].grid(True, alpha=0.3)
        if min(history["loss"]) > 0:
            axes[0].set_yscale('log')

        axes[1].plot(history["accuracy"], 'g-', linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])

        p_range = np.linspace(0.01, 0.99, 200)
        ce_loss_1 = -np.log(p_range)
        mse_loss_1 = (1 - p_range) ** 2

        axes[2].plot(p_range, ce_loss_1, 'b-', linewidth=2, label='CE (true=1)')
        axes[2].plot(p_range, mse_loss_1, 'r--', linewidth=2, label='MSE (true=1)')
        axes[2].set_xlabel("Predicted Probability")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("Loss Function Shapes")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Loss", f"{history['loss'][-1]:.4f}")
        col2.metric("Final Train Accuracy", f"{history['accuracy'][-1]:.3f}")
        col3.metric("Loss Reduction", f"{history['loss'][0] - history['loss'][-1]:.4f}")

    with tab6:
        st.subheader("Backpropagation & Gradient Descent")
        st.write("Backpropagation computes how much each weight contributed to the error, then gradient descent adjusts weights in the direction that reduces the loss. This is how the network learns.")
        st.info("**What to observe:** Gradients show how much each weight needs to change. Watch the weight distributions shift during training. Large gradients mean big updates, small ones mean the network is converging.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        snapshots = history["params_snapshots"]
        if len(snapshots) >= 2:
            early_params = snapshots[0][1]
            late_params = snapshots[-1][1]

            all_early_w = np.concatenate([early_params[k].flatten() for k in early_params if k.startswith("W")])
            all_late_w = np.concatenate([late_params[k].flatten() for k in late_params if k.startswith("W")])

            axes[0][0].hist(all_early_w, bins=50, alpha=0.7, color='blue', label=f'Epoch {snapshots[0][0]}')
            axes[0][0].hist(all_late_w, bins=50, alpha=0.7, color='red', label=f'Epoch {snapshots[-1][0]}')
            axes[0][0].set_title("Weight Distribution (Before vs After)")
            axes[0][0].set_xlabel("Weight Value")
            axes[0][0].set_ylabel("Count")
            axes[0][0].legend()

        grad_snapshots = history["grads_snapshots"]
        if grad_snapshots:
            final_grads = grad_snapshots[-1][1]
            layer_grad_norms = []
            layer_labels = []
            for k in sorted(final_grads.keys()):
                if k.startswith("dW"):
                    layer_grad_norms.append(np.linalg.norm(final_grads[k]))
                    layer_labels.append(k.replace("dW", "Layer "))
            axes[0][1].bar(layer_labels, layer_grad_norms, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:len(layer_labels)])
            axes[0][1].set_title("Gradient Magnitude per Layer (Final)")
            axes[0][1].set_ylabel("Gradient Norm")
            axes[0][1].grid(True, alpha=0.3, axis='y')

        grad_norms_over_time = []
        for epoch, grads in grad_snapshots:
            total_norm = np.sqrt(sum(np.sum(grads[k] ** 2) for k in grads if k.startswith("dW")))
            grad_norms_over_time.append((epoch, total_norm))

        if grad_norms_over_time:
            epochs_list, norms_list = zip(*grad_norms_over_time)
            axes[1][0].plot(epochs_list, norms_list, 'r-o', markersize=5)
            axes[1][0].set_title("Total Gradient Norm Over Training")
            axes[1][0].set_xlabel("Epoch")
            axes[1][0].set_ylabel("Gradient Norm")
            axes[1][0].grid(True, alpha=0.3)

        if show_gradients and grad_snapshots:
            final_grads = grad_snapshots[-1][1]
            grad_keys = [k for k in sorted(final_grads.keys()) if k.startswith("dW")]
            if grad_keys:
                all_grads = np.concatenate([final_grads[k].flatten() for k in grad_keys])
                axes[1][1].hist(all_grads, bins=50, color='purple', alpha=0.7)
                axes[1][1].set_title("Gradient Value Distribution (Final)")
                axes[1][1].set_xlabel("Gradient Value")
                axes[1][1].set_ylabel("Count")
        else:
            axes[1][1].text(0.5, 0.5, "Enable 'Show Gradients'\nin sidebar", ha='center', va='center', fontsize=14, transform=axes[1][1].transAxes)
            axes[1][1].set_title("Gradient Distribution")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.write("**Weight update rule:** W_new = W_old - learning_rate x gradient")
        st.write(f"**Current learning rate:** {learning_rate}")

        if show_gradients and grad_snapshots:
            st.write("**Per-layer gradient stats (final epoch):**")
            final_grads = grad_snapshots[-1][1]
            grad_stats = []
            for k in sorted(final_grads.keys()):
                if k.startswith("dW"):
                    g = final_grads[k]
                    grad_stats.append({
                        "Layer": k.replace("dW", ""),
                        "Mean": f"{np.mean(g):.6f}",
                        "Std": f"{np.std(g):.6f}",
                        "Max": f"{np.max(np.abs(g)):.6f}",
                        "Shape": str(g.shape),
                    })
            st.dataframe(grad_stats, use_container_width=True)

    with tab7:
        st.subheader("Model Complexity: Depth, Width & Overfitting")
        st.write("A network's capacity depends on its depth (layers) and width (neurons per layer). Too little capacity = underfitting. Too much = overfitting. Finding the right balance is key.")
        st.info("**What to observe:** Compare decision boundaries across different architectures. Simple networks make smooth boundaries, complex ones can fit any shape -- including noise.")

        configs = [
            ("1 layer, 2 neurons", [2, 2, 1]),
            ("1 layer, 8 neurons", [2, 8, 1]),
            ("2 layers, 8 neurons", [2, 8, 8, 1]),
            ("3 layers, 16 neurons", [2, 16, 16, 16, 1]),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        for idx, (cname, sizes) in enumerate(configs):
            n_h = len(sizes) - 2
            p, h, _ = train_network(X_train, Y_train, sizes, learning_rate, min(n_epochs, 500), activation_name, loss_name)
            plot_decision_boundary_nn(p, X_train, y_train, n_h, act_fn, cname, axes[idx])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.write("**Training vs Test Accuracy by Architecture:**")
        results = []
        for cname, sizes in configs:
            n_h = len(sizes) - 2
            p, h, _ = train_network(X_train, Y_train, sizes, learning_rate, min(n_epochs, 500), activation_name, loss_name)
            tr_pred = (predict(X_train, p, n_h, act_fn) >= 0.5).astype(int)
            te_pred = (predict(X_test, p, n_h, act_fn) >= 0.5).astype(int)
            tr_acc = np.mean(tr_pred == Y_train)
            te_acc = np.mean(te_pred == Y_test)
            total_p = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
            results.append({
                "Architecture": cname,
                "Parameters": total_p,
                "Train Acc": f"{tr_acc:.3f}",
                "Test Acc": f"{te_acc:.3f}",
                "Gap": f"{tr_acc - te_acc:.3f}",
            })
        st.dataframe(results, use_container_width=True)

        st.write("**Learning Rate Comparison:**")
        lr_values = [0.01, 0.1, 0.5, 1.0]
        fig2, axes2 = plt.subplots(1, len(lr_values), figsize=(5 * len(lr_values), 4))

        for idx, lr in enumerate(lr_values):
            p, h, _ = train_network(X_train, Y_train, layer_sizes, lr, min(n_epochs, 500), activation_name, loss_name)
            axes2[idx].plot(h["loss"], 'b-', linewidth=1)
            axes2[idx].set_title(f"LR = {lr}\nFinal loss: {h['loss'][-1]:.4f}")
            axes2[idx].set_xlabel("Epoch")
            axes2[idx].set_ylabel("Loss")
            axes2[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.sidebar.markdown("---")
    st.sidebar.metric("Train Accuracy", f"{final_train_acc:.3f}")
    st.sidebar.metric("Test Accuracy", f"{final_test_acc:.3f}")
    st.sidebar.metric("Final Loss", f"{history['loss'][-1]:.4f}")
