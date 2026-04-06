import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque


def plot_decision_boundary(clf, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', s=20)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def compute_impurity(y_subset, method="gini"):
    if len(y_subset) == 0:
        return 0.0
    classes, counts = np.unique(y_subset, return_counts=True)
    probs = counts / len(y_subset)
    if method == "gini":
        return 1.0 - np.sum(probs ** 2)
    else:
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))


def find_best_split(X_data, y_data, method="gini"):
    best_gain = -1
    best_feature = None
    best_threshold = None
    parent_impurity = compute_impurity(y_data, method)
    n = len(y_data)
    all_splits = []

    for feature_idx in range(X_data.shape[1]):
        thresholds = np.unique(X_data[:, feature_idx])
        for thresh in thresholds:
            left_mask = X_data[:, feature_idx] <= thresh
            right_mask = ~left_mask
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            left_impurity = compute_impurity(y_data[left_mask], method)
            right_impurity = compute_impurity(y_data[right_mask], method)
            weighted_impurity = (np.sum(left_mask) / n) * left_impurity + (np.sum(right_mask) / n) * right_impurity
            gain = parent_impurity - weighted_impurity
            all_splits.append((feature_idx, thresh, gain, left_impurity, right_impurity))
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = thresh

    return best_feature, best_threshold, best_gain, all_splits


def get_prediction_path(tree, point):
    tree_ = tree.tree_
    path = []
    node = 0
    while tree_.children_left[node] != tree_.children_right[node]:
        feat = tree_.feature[node]
        thresh = tree_.threshold[node]
        if point[feat] <= thresh:
            direction = "left"
            path.append((node, feat, thresh, point[feat], direction))
            node = tree_.children_left[node]
        else:
            direction = "right"
            path.append((node, feat, thresh, point[feat], direction))
            node = tree_.children_right[node]
    predicted_class = np.argmax(tree_.value[node][0])
    return path, predicted_class, node


def run():
    st.title("Decision Tree Explorer")

    st.sidebar.header("Decision Tree Controls")

    dataset_name = st.sidebar.selectbox("Dataset", ["Simple (Linearly Separable)", "Moons (Non-linear)", "Circles (Concentric)", "Noisy"], key="dt_dataset")
    max_depth = st.sidebar.slider("Max Depth", 1, 15, 3, key="dt_depth")
    min_samples_split = st.sidebar.slider("Min Samples per Split", 2, 50, 2, key="dt_min_samples")
    criterion = st.sidebar.selectbox("Impurity Measure", ["gini", "entropy"], key="dt_criterion")
    add_noise = st.sidebar.toggle("Add Noise to Data", False, key="dt_noise")
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.3, step=0.05, key="dt_noise_level") if add_noise else 0.0
    show_impurity = st.sidebar.toggle("Show Impurity Values", True, key="dt_show_imp")
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20, key="dt_test_size")

    np.random.seed(42)
    n_samples = 300

    if dataset_name == "Simple (Linearly Separable)":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
    elif dataset_name == "Moons (Non-linear)":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif dataset_name == "Circles (Concentric)":
        X, y = make_circles(n_samples=n_samples, noise=0.15, factor=0.5, random_state=42)
    else:
        X, y = make_moons(n_samples=n_samples, noise=0.5, random_state=42)

    if add_noise:
        X = X + np.random.randn(*X.shape) * noise_level

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    def get_split_lines(tree):
        lines = []
        tree_ = tree.tree_

        def recurse(node, depth, x_min, x_max, y_min, y_max):
            if tree_.feature[node] == -2:
                return
            feat = tree_.feature[node]
            thresh = tree_.threshold[node]
            if feat == 0:
                lines.append(("v", thresh, y_min, y_max, depth))
                recurse(tree_.children_left[node], depth + 1, x_min, thresh, y_min, y_max)
                recurse(tree_.children_right[node], depth + 1, thresh, x_max, y_min, y_max)
            else:
                lines.append(("h", thresh, x_min, x_max, depth))
                recurse(tree_.children_left[node], depth + 1, x_min, x_max, y_min, thresh)
                recurse(tree_.children_right[node], depth + 1, x_min, x_max, thresh, y_max)

        xmn, xmx = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        ymn, ymx = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        recurse(0, 0, xmn, xmx, ymn, ymx)
        return lines

    def build_tree_data(tree):
        tree_ = tree.tree_
        nodes = []

        def recurse(node_id, depth, parent_id=None, is_left=None):
            is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
            samples = tree_.n_node_samples[node_id]
            values = tree_.value[node_id][0]
            cls = np.argmax(values)
            impurity = tree_.impurity[node_id]

            label_parts = []
            if not is_leaf:
                feat = tree_.feature[node_id]
                thresh = tree_.threshold[node_id]
                label_parts.append(f"Feature {feat + 1} <= {thresh:.2f}")
            label_parts.append(f"Samples: {samples}")
            label_parts.append(f"Class: {cls}")
            if show_impurity:
                label_parts.append(f"{criterion}: {impurity:.3f}")

            nodes.append({
                "id": node_id, "depth": depth, "parent": parent_id,
                "is_left": is_left, "is_leaf": is_leaf,
                "label": "\n".join(label_parts), "class": cls,
                "impurity": impurity, "samples": samples,
            })

            if not is_leaf:
                recurse(tree_.children_left[node_id], depth + 1, node_id, True)
                recurse(tree_.children_right[node_id], depth + 1, node_id, False)

        recurse(0, 0)
        return nodes

    def plot_tree_plotly(tree):
        nodes = build_tree_data(tree)
        node_map = {n["id"]: n for n in nodes}

        positions = {}
        level_counts = {}
        for n in nodes:
            d = n["depth"]
            level_counts[d] = level_counts.get(d, 0) + 1

        max_depth_val = max(n["depth"] for n in nodes)
        level_idx = {}

        queue = deque()
        queue.append(nodes[0]["id"])
        while queue:
            nid = queue.popleft()
            node = node_map[nid]
            d = node["depth"]
            idx = level_idx.get(d, 0)
            level_idx[d] = idx + 1
            total_at_level = level_counts[d]
            x = (idx + 0.5) / total_at_level
            ypos = 1.0 - d / (max_depth_val + 1)
            positions[nid] = (x, ypos)

            tree_ = tree.tree_
            if tree_.children_left[nid] != tree_.children_right[nid]:
                queue.append(tree_.children_left[nid])
                queue.append(tree_.children_right[nid])

        edge_x, edge_y = [], []
        for n in nodes:
            if n["parent"] is not None:
                px, py = positions[n["parent"]]
                cx, cy = positions[n["id"]]
                edge_x += [px, cx, None]
                edge_y += [py, cy, None]

        node_x = [positions[n["id"]][0] for n in nodes]
        node_y = [positions[n["id"]][1] for n in nodes]
        node_colors = ["#ff6b6b" if n["class"] == 0 else "#4ecdc4" for n in nodes]
        node_text = [n["label"] for n in nodes]
        node_sizes = [max(15, min(40, n["samples"] // 3)) for n in nodes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#888', width=1.5), hoverinfo='none'))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                 marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='black')),
                                 text=[f"{'Leaf' if n['is_leaf'] else 'Node'} {n['id']}" for n in nodes],
                                 textposition="top center", hovertext=node_text, hoverinfo='text'))
        fig.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          height=400 + max_depth_val * 80, margin=dict(l=20, r=20, t=20, b=20))
        return fig

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Data & Splits", "Impurity Measures", "Split Selection",
        "Tree Growth", "Prediction Path", "Overfitting & Depth", "Noise & Pruning"
    ])

    with tab1:
        st.subheader("Data Partitioning & Feature Splits")
        st.write("The Decision Tree splits the 2D space into rectangular regions. Each colored region represents a different predicted class. The lines show where the tree made its decisions.")
        st.info("**What to observe:** Notice how the tree creates axis-aligned rectangular boundaries. Deeper trees create more, smaller rectangles to fit the data better.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='k', s=20)
        axes[0].set_title("Raw Training Data")
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")

        plot_decision_boundary(clf, X_train, y_train, f"Decision Regions (depth={max_depth})", axes[1])

        split_lines = get_split_lines(clf)
        for line_type, val, low, high, depth in split_lines:
            alpha = max(0.2, 1.0 - depth * 0.15)
            lw = max(0.5, 2.5 - depth * 0.3)
            if line_type == "v":
                axes[1].plot([val, val], [low, high], 'k-', alpha=alpha, linewidth=lw)
            else:
                axes[1].plot([low, high], [val, val], 'k-', alpha=alpha, linewidth=lw)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Splits", len(split_lines))
        col2.metric("Tree Depth", clf.get_depth())
        col3.metric("Leaf Nodes", clf.get_n_leaves())

    with tab2:
        st.subheader("Impurity Measures: Gini vs Entropy")
        st.write("Impurity tells us how mixed the classes are at a node. A pure node (all one class) has impurity = 0. The tree picks splits that reduce impurity the most.")
        st.info("**What to observe:** Both Gini and Entropy peak when classes are evenly mixed (50/50). They just have slightly different shapes — Gini is bounded by 0.5, Entropy by 1.0.")

        p_range = np.linspace(0.001, 0.999, 200)
        gini_vals = 2 * p_range * (1 - p_range)
        entropy_vals = -(p_range * np.log2(p_range) + (1 - p_range) * np.log2(1 - p_range))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(p_range, gini_vals, 'b-', linewidth=2, label='Gini Impurity')
        axes[0].plot(p_range, entropy_vals, 'r-', linewidth=2, label='Entropy')
        axes[0].set_xlabel("Proportion of Class 1")
        axes[0].set_ylabel("Impurity")
        axes[0].set_title("Gini vs Entropy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        tree_ = clf.tree_
        node_impurities = []
        for i in range(tree_.node_count):
            node_impurities.append(tree_.impurity[i])

        axes[1].bar(range(len(node_impurities)), node_impurities, color=['#ff6b6b' if imp > 0.3 else '#4ecdc4' for imp in node_impurities])
        axes[1].set_xlabel("Node Index")
        axes[1].set_ylabel(f"{criterion.capitalize()} Impurity")
        axes[1].set_title(f"Impurity at Each Node ({criterion.capitalize()})")
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        parent_imp = compute_impurity(y_train, criterion)
        st.write(f"**Root node {criterion} impurity:** {parent_imp:.4f}")

        class_counts = np.bincount(y_train)
        cols = st.columns(len(class_counts))
        for i, count in enumerate(class_counts):
            cols[i].metric(f"Class {i} samples", count)

    with tab3:
        st.subheader("Split Selection: Finding the Best Feature & Threshold")
        st.write("At each node, the tree tries every possible feature and threshold. It picks the one that gives the biggest impurity reduction (information gain).")
        st.info("**What to observe:** The highlighted split gives the maximum gain. Try changing the impurity measure in the sidebar to see how the best split might differ.")

        best_feat, best_thresh, best_gain, all_splits = find_best_split(X_train, y_train, criterion)

        feat1_splits = [(t, g) for f, t, g, li, ri in all_splits if f == 0]
        feat2_splits = [(t, g) for f, t, g, li, ri in all_splits if f == 1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if feat1_splits:
            t1, g1 = zip(*feat1_splits)
            axes[0].plot(t1, g1, 'b-', alpha=0.7, linewidth=1)
            axes[0].scatter(t1, g1, c='blue', s=10, alpha=0.5)
        if feat2_splits:
            t2, g2 = zip(*feat2_splits)
            axes[0].plot(t2, g2, 'r-', alpha=0.7, linewidth=1)
            axes[0].scatter(t2, g2, c='red', s=10, alpha=0.5)

        if best_feat is not None:
            axes[0].axvline(x=best_thresh, color='green', linestyle='--', linewidth=2, label=f'Best: F{best_feat + 1}={best_thresh:.2f}')
        axes[0].set_xlabel("Threshold Value")
        axes[0].set_ylabel("Information Gain")
        axes[0].set_title("Gain for All Candidate Splits (Root Node)")
        axes[0].legend(["Feature 1", "Feature 2", "Best Split"])
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='k', s=20)
        if best_feat == 0:
            axes[1].axvline(x=best_thresh, color='green', linestyle='--', linewidth=2.5, label=f'Best Split: F1 <= {best_thresh:.2f}')
        else:
            axes[1].axhline(y=best_thresh, color='green', linestyle='--', linewidth=2.5, label=f'Best Split: F2 <= {best_thresh:.2f}')
        axes[1].set_title("Best Split on Data")
        axes[1].set_xlabel("Feature 1")
        axes[1].set_ylabel("Feature 2")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2, col3 = st.columns(3)
        col1.metric("Best Feature", f"Feature {best_feat + 1}" if best_feat is not None else "N/A")
        col2.metric("Best Threshold", f"{best_thresh:.4f}" if best_thresh is not None else "N/A")
        col3.metric("Best Gain", f"{best_gain:.4f}")

        top_splits = sorted(all_splits, key=lambda x: -x[2])[:10]
        st.write("**Top 10 Candidate Splits:**")
        split_data = []
        for f, t, g, li, ri in top_splits:
            split_data.append({"Feature": f"Feature {f + 1}", "Threshold": f"{t:.4f}", "Gain": f"{g:.4f}",
                               "Left Impurity": f"{li:.4f}", "Right Impurity": f"{ri:.4f}"})
        st.dataframe(split_data, use_container_width=True)

    with tab4:
        st.subheader("Tree Structure & Growth")
        st.write("The tree grows by recursively splitting nodes. Starting from the root (all data), each internal node tests a condition, and leaf nodes give final predictions.")
        st.info("**What to observe:** Hover over nodes to see their stats. Larger nodes have more samples. Red = Class 0, Teal = Class 1.")

        st.plotly_chart(plot_tree_plotly(clf), use_container_width=True)

        st.write("**Tree Rules (Text Format):**")
        tree_text = export_text(clf, feature_names=["Feature_1", "Feature_2"], max_depth=10)
        st.code(tree_text)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Depth", clf.get_depth())
        col2.metric("Leaves", clf.get_n_leaves())
        col3.metric("Total Nodes", clf.tree_.node_count)
        col4.metric("Features Used", len(set(clf.tree_.feature[clf.tree_.feature >= 0])))

    with tab5:
        st.subheader("Prediction Path (Decision Rules)")
        st.write("For any data point, the tree follows a path from root to leaf. Each step checks one feature against a threshold. The leaf node gives the final prediction.")
        st.info("**What to observe:** Move the sliders to see how the tree traces different paths. The rules are simple if-else chains.")

        col1, col2 = st.columns(2)
        with col1:
            px = st.slider("Point Feature 1", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()), key="dt_px")
        with col2:
            py = st.slider("Point Feature 2", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()), key="dt_py")

        test_point = np.array([px, py])
        path, pred_class, leaf_node = get_prediction_path(clf, test_point)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plot_decision_boundary(clf, X_train, y_train, "Prediction Path", ax)
        ax.scatter([px], [py], c='yellow', s=200, marker='*', edgecolors='black', linewidths=2, zorder=5, label=f'Query Point -> Class {pred_class}')
        ax.legend(fontsize=10)
        st.pyplot(fig)
        plt.close()

        st.write("**Decision Path:**")
        for i, (node, feat, thresh, val, direction) in enumerate(path):
            symbol = "<=" if direction == "left" else ">"
            st.write(f"**Step {i + 1}:** Node {node} -- Feature {feat + 1} = {val:.3f} {symbol} {thresh:.3f} -> go **{direction}**")
        st.success(f"**Prediction: Class {pred_class}** (reached leaf node {leaf_node})")

    with tab6:
        st.subheader("Overfitting & Depth Analysis")
        st.write("Deeper trees fit training data better but may memorize noise. The gap between training and test accuracy reveals overfitting.")
        st.info("**What to observe:** Watch training accuracy climb toward 100% while test accuracy may plateau or drop. The sweet spot is where test accuracy peaks.")

        depths = range(1, 16)
        train_accs = []
        test_accs = []
        leaf_counts = []

        for d in depths:
            temp_clf = DecisionTreeClassifier(max_depth=d, min_samples_split=min_samples_split, criterion=criterion, random_state=42)
            temp_clf.fit(X_train, y_train)
            train_accs.append(accuracy_score(y_train, temp_clf.predict(X_train)))
            test_accs.append(accuracy_score(y_test, temp_clf.predict(X_test)))
            leaf_counts.append(temp_clf.get_n_leaves())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(list(depths), train_accs, 'b-o', label='Train Accuracy', markersize=5)
        axes[0].plot(list(depths), test_accs, 'r-o', label='Test Accuracy', markersize=5)
        axes[0].axvline(x=max_depth, color='green', linestyle='--', alpha=0.7, label=f'Current depth={max_depth}')
        axes[0].set_xlabel("Max Depth")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Training vs Test Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(list(depths), leaf_counts, 'g-o', markersize=5)
        axes[1].set_xlabel("Max Depth")
        axes[1].set_ylabel("Number of Leaves")
        axes[1].set_title("Tree Complexity")
        axes[1].grid(True, alpha=0.3)

        axes[2].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        compare_depths = [1, 3, 7, 15]
        fig2, axes2 = plt.subplots(1, 4, figsize=(20, 4))
        for idx, d in enumerate(compare_depths):
            temp_clf = DecisionTreeClassifier(max_depth=d, criterion=criterion, random_state=42)
            temp_clf.fit(X_train, y_train)
            plot_decision_boundary(temp_clf, X_train, y_train, f"Depth = {d}", axes2[idx])
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        gap = [train_accs[i] - test_accs[i] for i in range(len(list(depths)))]
        best_depth_idx = np.argmax(test_accs)
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Test Accuracy", f"{max(test_accs):.3f}")
        col2.metric("Best Depth", best_depth_idx + 1)
        col3.metric("Overfit Gap at Current Depth", f"{gap[max_depth - 1]:.3f}")

    with tab7:
        st.subheader("Effect of Noise & Pruning")
        st.write("Noise makes data harder to separate cleanly. Deep trees will try to fit every noisy point, creating jagged boundaries. Limiting depth acts as a simple form of pruning.")
        st.info("**What to observe:** Compare how a deep tree creates complex boundaries on noisy data, while a shallow tree stays smooth. The shallow tree often generalizes better.")

        noise_levels_compare = [0.0, 0.2, 0.5, 0.8]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for idx, nl in enumerate(noise_levels_compare):
            X_noisy, y_noisy = make_moons(n_samples=n_samples, noise=0.2 + nl, random_state=42)
            X_n_train, X_n_test, y_n_train, y_n_test = train_test_split(X_noisy, y_noisy, test_size=0.2, random_state=42)

            deep_clf = DecisionTreeClassifier(max_depth=15, criterion=criterion, random_state=42)
            deep_clf.fit(X_n_train, y_n_train)
            deep_test_acc = accuracy_score(y_n_test, deep_clf.predict(X_n_test))
            plot_decision_boundary(deep_clf, X_noisy, y_noisy, f"Deep (noise={nl})\nTest acc: {deep_test_acc:.2f}", axes[0][idx])

            shallow_clf = DecisionTreeClassifier(max_depth=3, criterion=criterion, random_state=42)
            shallow_clf.fit(X_n_train, y_n_train)
            shallow_test_acc = accuracy_score(y_n_test, shallow_clf.predict(X_n_test))
            plot_decision_boundary(shallow_clf, X_noisy, y_noisy, f"Shallow (noise={nl})\nTest acc: {shallow_test_acc:.2f}", axes[1][idx])

        axes[0][0].set_ylabel("Deep Tree (depth=15)", fontsize=12)
        axes[1][0].set_ylabel("Shallow Tree (depth=3)", fontsize=12)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.write("**Pruning Comparison (Current Dataset):**")

        pruning_depths = [2, 3, 5, 10, None]
        pruning_results = []
        for d in pruning_depths:
            temp_clf = DecisionTreeClassifier(max_depth=d, min_samples_split=min_samples_split, criterion=criterion, random_state=42)
            temp_clf.fit(X_train, y_train)
            tr_acc = accuracy_score(y_train, temp_clf.predict(X_train))
            te_acc = accuracy_score(y_test, temp_clf.predict(X_test))
            pruning_results.append({
                "Max Depth": str(d) if d else "None (unlimited)",
                "Train Accuracy": f"{tr_acc:.3f}",
                "Test Accuracy": f"{te_acc:.3f}",
                "Leaves": temp_clf.get_n_leaves(),
                "Overfit Gap": f"{tr_acc - te_acc:.3f}"
            })
        st.dataframe(pruning_results, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.metric("Train Accuracy", f"{train_acc:.3f}")
    st.sidebar.metric("Test Accuracy", f"{test_acc:.3f}")
