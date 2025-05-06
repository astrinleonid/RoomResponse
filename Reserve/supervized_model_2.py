import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import itertools


def build_supervised_model(dataset_path='mfcc_dataset.csv'):
    """
    Build and evaluate supervised learning models on the MFCC dataset.

    Args:
        dataset_path (str): Path to the CSV file containing the MFCC dataset

    Returns:
        tuple: (best_models, X_test, y_test, feature_cols, X, y, scaler)
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} columns")

    # Extract features and labels
    feature_cols = [col for col in df.columns if col.startswith('mfcc_')]
    X = df[feature_cols].values
    y = df['label'].values

    # Create a scaler for later use in visualizations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create classification models
    models = {
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True))
        ]),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ]),
        'Neural Network': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(random_state=42, max_iter=1000))
        ])
    }

    # Define parameter grids for GridSearchCV
    param_grids = {
        'SVM': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.1, 0.01],
            'classifier__kernel': ['rbf', 'linear']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        },
        'Neural Network': {
            'classifier__hidden_layer_sizes': [(10,), (20,), (10, 10)],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__activation': ['relu', 'tanh']
        }
    }

    # Perform cross-validation and hyperparameter tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_models = {}
    cv_results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Perform Grid Search
        grid_search = GridSearchCV(
            model, param_grids[name], cv=cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        cv_results[name] = grid_search.cv_results_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

        # Evaluate on test set
        y_pred = grid_search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")

        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    # Compare models
    compare_models(best_models, X_test, y_test)

    # Feature importance analysis for Random Forest
    if 'Random Forest' in best_models:
        rf_model = best_models['Random Forest']
        feature_importance = feature_importance_analysis(rf_model, feature_cols, X_test, y_test)

        # Plot projections of data on important features
        plot_feature_projections(X_scaled, y, feature_cols, feature_importance)

    return best_models, X_test, y_test, feature_cols, X, y, scaler


def feature_importance_analysis(rf_model, feature_cols, X_test, y_test):
    """
    Analyze feature importance from the Random Forest model.

    Returns:
        list: Feature importances sorted by importance (tuples of feature index and importance)
    """
    # Get feature importances
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
    else:
        # If the model is in a pipeline, extract the classifier
        if hasattr(rf_model, 'named_steps') and 'classifier' in rf_model.named_steps:
            importances = rf_model.named_steps['classifier'].feature_importances_
        else:
            print("Could not extract feature importances from the model")
            return []

    # Sort feature importances
    sorted_indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[sorted_indices], align='center')
    plt.xticks(range(len(importances)), [feature_cols[i] for i in sorted_indices], rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)

    # Cumulative importance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(importances) + 1), np.cumsum(importances[sorted_indices]), 'o-')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.grid(True)
    plt.savefig('cumulative_importance.png', dpi=300)
    plt.show()

    # Print top 5 features
    print("\nTop 5 Important Features:")
    for i in range(min(5, len(sorted_indices))):
        print(f"{feature_cols[sorted_indices[i]]}: {importances[sorted_indices[i]]:.4f}")

    # Create list of (feature_index, importance) tuples
    feature_importance = [(i, importances[i]) for i in sorted_indices]
    return feature_importance


def plot_feature_projections(X, y, feature_cols, feature_importance):
    """
    Plot projections of the data onto the most important features.

    Args:
        X (numpy.ndarray): Scaled feature data
        y (numpy.ndarray): Labels
        feature_cols (list): Names of features
        feature_importance (list): List of (feature_index, importance) tuples, sorted by importance
    """
    if not feature_importance:
        print("No feature importance available. Cannot create projections.")
        return

    # Get indices of the top 5 most important features
    top_feature_indices = [idx for idx, _ in feature_importance[:5]]

    # 1. Pairwise scatter plots of top features
    plot_pairwise_features(X, y, top_feature_indices, feature_cols)

    # 2. 3D scatter plot of top 3 features
    plot_3d_features(X, y, top_feature_indices[:3], feature_cols)

    # 3. Parallel coordinates plot of top features
    plot_parallel_coordinates(X, y, top_feature_indices, feature_cols)

    # 4. Andrews curves
    plot_andrews_curves(X, y, top_feature_indices, feature_cols)

    # 5. Distribution plots
    plot_distributions(X, y, top_feature_indices, feature_cols)


def plot_pairwise_features(X, y, feature_indices, feature_cols):
    """
    Create pairwise scatter plots of the most important features.
    """
    n_features = len(feature_indices)
    if n_features < 2:
        print("Not enough features for pairwise plots.")
        return

    # Limit to at most 4 features for readability
    n_features = min(n_features, 4)
    feature_indices = feature_indices[:n_features]

    # Create a grid of pairwise scatter plots
    fig, axes = plt.subplots(n_features, n_features, figsize=(3 * n_features, 3 * n_features))

    for i, idx1 in enumerate(feature_indices):
        for j, idx2 in enumerate(feature_indices):
            if i == j:  # Diagonal plots - show histograms by class
                for label in np.unique(y):
                    axes[i, j].hist(X[y == label, idx1], alpha=0.5, label=f'Class {label}')
                axes[i, j].set_title(feature_cols[idx1])
                if i == n_features - 1:  # Only show legend in the last diagonal plot
                    axes[i, j].legend()
            else:  # Off-diagonal plots - show scatter plots
                for label in np.unique(y):
                    axes[i, j].scatter(
                        X[y == label, idx2],
                        X[y == label, idx1],
                        alpha=0.7,
                        label=f'Class {label}'
                    )
                if i == n_features - 1:  # Bottom row - add x labels
                    axes[i, j].set_xlabel(feature_cols[idx2])
                if j == 0:  # Leftmost column - add y labels
                    axes[i, j].set_ylabel(feature_cols[idx1])

    plt.tight_layout()
    plt.savefig('pairwise_features.png', dpi=300)
    plt.show()


def plot_3d_features(X, y, feature_indices, feature_cols):
    """
    Create a 3D scatter plot of the top 3 features.
    """
    if len(feature_indices) < 3:
        print("Not enough features for 3D plot.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for label in np.unique(y):
        ax.scatter(
            X[y == label, feature_indices[0]],
            X[y == label, feature_indices[1]],
            X[y == label, feature_indices[2]],
            label=f'Class {label}',
            alpha=0.7,
            s=50
        )

    ax.set_xlabel(feature_cols[feature_indices[0]])
    ax.set_ylabel(feature_cols[feature_indices[1]])
    ax.set_zlabel(feature_cols[feature_indices[2]])
    ax.set_title('3D Projection of Top 3 Features')
    ax.legend()

    plt.tight_layout()
    plt.savefig('3d_features.png', dpi=300)
    plt.show()


def plot_parallel_coordinates(X, y, feature_indices, feature_cols):
    """
    Create a parallel coordinates plot of the top features.
    """
    if len(feature_indices) < 2:
        print("Not enough features for parallel coordinates plot.")
        return

    # Create a DataFrame with selected features and label
    df = pd.DataFrame(X[:, feature_indices], columns=[feature_cols[i] for i in feature_indices])
    df['label'] = y

    plt.figure(figsize=(12, 8))
    pd.plotting.parallel_coordinates(df, 'label', colormap='viridis')
    plt.title('Parallel Coordinates Plot of Top Features')
    plt.tight_layout()
    plt.savefig('parallel_coordinates.png', dpi=300)
    plt.show()


def plot_andrews_curves(X, y, feature_indices, feature_cols):
    """
    Create Andrews curves of the top features.
    """
    if len(feature_indices) < 2:
        print("Not enough features for Andrews curves.")
        return

    # Create a DataFrame with selected features and label
    df = pd.DataFrame(X[:, feature_indices], columns=[feature_cols[i] for i in feature_indices])
    df['label'] = y

    plt.figure(figsize=(12, 8))
    pd.plotting.andrews_curves(df, 'label', colormap='viridis')
    plt.title('Andrews Curves of Top Features')
    plt.tight_layout()
    plt.savefig('andrews_curves.png', dpi=300)
    plt.show()


def plot_distributions(X, y, feature_indices, feature_cols):
    """
    Plot distribution of top features for each class.
    """
    n_features = len(feature_indices)
    classes = np.unique(y)
    n_classes = len(classes)

    if n_features == 0:
        print("No features available for distribution plots.")
        return

    # Create a figure with subplots for each feature
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    for i, idx in enumerate(feature_indices):
        for cls in classes:
            sns.kdeplot(X[y == cls, idx], ax=axes[i], label=f'Class {cls}')

        axes[i].set_title(f'Distribution of {feature_cols[idx]}')
        axes[i].set_xlabel(feature_cols[idx])
        axes[i].set_ylabel('Density')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300)
    plt.show()

def compare_models(models, X_test, y_test):
    """
    Compare the performance of different models on the test set.
    """
    # Plot confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))

    # If only one model, make axes iterable
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix: {name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig('model_comparison_cm.png', dpi=300)

    # Plot ROC curves
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            except:
                print(f"Could not calculate ROC curve for {name}")

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('model_comparison_roc.png', dpi=300)

    # Compare accuracy
    model_scores = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_scores[name] = accuracy

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_scores.keys(), model_scores.values())

    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center')

    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.savefig('model_comparison_accuracy.png', dpi=300)
    plt.show()


# Main execution
if __name__ == "__main__":
    best_models, X_test, y_test, feature_cols, X, y, scaler = build_supervised_model('mfcc_dataset.csv')

    # Get the best model
    model_accuracies = {}
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        model_accuracies[name] = accuracy_score(y_test, y_pred)

    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_model = best_models[best_model_name]

    print(f"\nBest model: {best_model_name} with accuracy: {model_accuracies[best_model_name]:.4f}")