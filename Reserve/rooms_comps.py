import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_room_discrimination(dataset_path='mfcc_dataset_with_metadata.csv'):
    """
    Analyze how well the model discriminates between different rooms.

    Args:
        dataset_path (str): Path to the CSV file containing the MFCC dataset with metadata

    Returns:
        dict: Dictionary containing classification results for rooms
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} columns")

    # Extract features
    feature_cols = [col for col in df.columns if col.startswith('mfcc_')]

    # Get unique rooms
    rooms = df['room_name'].unique()
    print(f"Found {len(rooms)} unique rooms: {rooms}")

    # Check if we have enough rooms for classification
    if len(rooms) < 2:
        print("Not enough different rooms for classification analysis.")
        return {}

    # Extract features and labels
    X = df[feature_cols].values
    y = pd.Categorical(df['room_name']).codes  # Convert room names to numeric codes

    # Create a mapping from numeric codes to room names for reference
    room_mapping = {code: room for code, room in enumerate(pd.Categorical(df['room_name']).categories)}

    print("\nRoom distribution:")
    for code, room in room_mapping.items():
        count = np.sum(y == code)
        print(f"  {room}: {count} samples (Code: {code})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM model
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nRoom classification test accuracy: {accuracy:.4f}")

    # Cross-validation for more robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm, X, y, cv=cv)

    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Classification report
    class_names = [room_mapping[i] for i in range(len(room_mapping))]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Room')
    plt.ylabel('True Room')
    plt.title('Room Classification Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('room_classification_confusion_matrix.png', dpi=300)
    plt.show()

    # Analyze which rooms are most confusable
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_normalized, 0)  # Zero out the diagonal for better visualization

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Room')
    plt.ylabel('True Room')
    plt.title('Room Confusion (Normalized)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('room_confusion_normalized.png', dpi=300)
    plt.show()

    return {
        'accuracy': accuracy,
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': cm,
        'room_mapping': room_mapping
    }


def analyze_computer_discrimination(dataset_path='mfcc_dataset_with_metadata.csv'):
    """
    Analyze how well the model discriminates between different computers.

    Args:
        dataset_path (str): Path to the CSV file containing the MFCC dataset with metadata

    Returns:
        dict: Dictionary containing classification results for computers
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} columns")

    # Extract features
    feature_cols = [col for col in df.columns if col.startswith('mfcc_')]

    # Get unique computers
    computers = df['computer_name'].unique()
    print(f"Found {len(computers)} unique computers: {computers}")

    # Check if we have enough computers for classification
    if len(computers) < 2:
        print("Not enough different computers for classification analysis.")
        return {}

    # Extract features and labels
    X = df[feature_cols].values
    y = pd.Categorical(df['computer_name']).codes  # Convert computer names to numeric codes

    # Create a mapping from numeric codes to computer names for reference
    computer_mapping = {code: computer for code, computer in enumerate(pd.Categorical(df['computer_name']).categories)}

    print("\nComputer distribution:")
    for code, computer in computer_mapping.items():
        count = np.sum(y == code)
        print(f"  {computer}: {count} samples (Code: {code})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM model
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nComputer classification test accuracy: {accuracy:.4f}")

    # Cross-validation for more robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm, X, y, cv=cv)

    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Classification report
    class_names = [computer_mapping[i] for i in range(len(computer_mapping))]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Computer')
    plt.ylabel('True Computer')
    plt.title('Computer Classification Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('computer_classification_confusion_matrix.png', dpi=300)
    plt.show()

    # Analyze which computers are most confusable
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_normalized, 0)  # Zero out the diagonal for better visualization

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Computer')
    plt.ylabel('True Computer')
    plt.title('Computer Confusion (Normalized)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('computer_confusion_normalized.png', dpi=300)
    plt.show()

    return {
        'accuracy': accuracy,
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': cm,
        'computer_mapping': computer_mapping
    }


def analyze_combined_factors(dataset_path='mfcc_dataset_with_metadata.csv'):
    """
    Analyze the relative importance of room, computer, and scenario in discrimination.

    Args:
        dataset_path (str): Path to the CSV file containing the MFCC dataset with metadata
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Extract features
    feature_cols = [col for col in df.columns if col.startswith('mfcc_')]
    X = df[feature_cols].values

    # Create different target variables
    targets = {
        'room': pd.Categorical(df['room_name']).codes,
        'computer': pd.Categorical(df['computer_name']).codes,
        'scenario': pd.Categorical(df['scenario_id']).codes
    }

    # Train and evaluate models for each target
    results = {}

    for target_name, y in targets.items():
        print(f"\nAnalyzing classification by {target_name.upper()}")

        # Check class distribution
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"  Not enough classes for {target_name} classification. Skipping.")
            continue

        # Count samples per class
        for cls in unique_classes:
            count = np.sum(y == cls)
            print(f"  Class {cls}: {count} samples")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

        try:
            cv_scores = cross_val_score(svm, X_scaled, y, cv=cv)

            print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            results[target_name] = {
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        except Exception as e:
            print(f"  Error in cross-validation: {e}")

    # Compare the results
    if results:
        # Create bar chart comparing accuracies
        plt.figure(figsize=(10, 6))
        labels = list(results.keys())
        accuracies = [results[label]['cv_accuracy'] for label in labels]
        errors = [results[label]['cv_std'] for label in labels]

        bars = plt.bar(labels, accuracies, yerr=errors, capsize=10)

        # Add accuracy values on top of bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{acc:.4f}', ha='center')

        plt.ylim(0, 1.1)
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Classification Accuracy by Different Factors')
        plt.tight_layout()
        plt.savefig('factor_comparison.png', dpi=300)
        plt.show()

        # Create a summary table
        summary = pd.DataFrame([
            {'Factor': factor, 'CV Accuracy': res['cv_accuracy'], 'CV Std': res['cv_std']}
            for factor, res in results.items()
        ])
        summary = summary.sort_values('CV Accuracy', ascending=False)

        print("\nComparison of Factors:")
        print(summary)
        summary.to_csv('factor_comparison.csv', index=False)

    return results


# Main execution
if __name__ == "__main__":
    dataset_path = 'mfcc_dataset_with_metadata.csv'

    # Analyze room discrimination
    print("\n" + "=" * 50)
    print("ROOM DISCRIMINATION ANALYSIS")
    print("=" * 50)
    room_results = analyze_room_discrimination(dataset_path)

    # Analyze computer discrimination
    print("\n" + "=" * 50)
    print("COMPUTER DISCRIMINATION ANALYSIS")
    print("=" * 50)
    computer_results = analyze_computer_discrimination(dataset_path)

    # Analyze combined factors
    print("\n" + "=" * 50)
    print("COMBINED FACTORS ANALYSIS")
    print("=" * 50)
    factor_results = analyze_combined_factors(dataset_path)

    # Print overall summary
    print("\n" + "=" * 50)
    print("OVERALL SUMMARY")
    print("=" * 50)

    print("\nDiscrimination accuracy by factor:")
    if 'room' in factor_results:
        print(f"  Room: {factor_results['room']['cv_accuracy']:.4f} ± {factor_results['room']['cv_std']:.4f}")
    if 'computer' in factor_results:
        print(
            f"  Computer: {factor_results['computer']['cv_accuracy']:.4f} ± {factor_results['computer']['cv_std']:.4f}")
    if 'scenario' in factor_results:
        print(
            f"  Scenario: {factor_results['scenario']['cv_accuracy']:.4f} ± {factor_results['scenario']['cv_std']:.4f}")