import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from ScenarioSelector import create_checkbox_interface



def pairwise_scenario_classification(df):
    """
    Build SVM models to discriminate between each pair of scenarios.

    Args:
        dataset_path (str): Path to the CSV file containing the MFCC dataset with metadata

    Returns:
        dict: Dictionary containing accuracy results for each scenario pair

    """


    print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} columns")

    df = df.sample(frac=1).reset_index(drop=True)
    # Extract features
    feature_cols = [col for col in df.columns if col.startswith('mfcc_')]

    # Get unique scenarios
    scenarios = df['scenario_id'].unique()
    print(f"Found {len(scenarios)} unique scenarios: {scenarios}")

    # Store results
    results = {}

    # Create a figure for the heatmap

    accuracy_matrix = np.zeros((len(scenarios), len(scenarios)))

    # For each pair of scenarios
    for i, j in itertools.combinations(range(len(scenarios)), 2):
        scenario1 = scenarios[i]
        scenario2 = scenarios[j]

        print(f"\nClassifying between Scenario {scenario1} and Scenario {scenario2}")

        # Filter data for these two scenarios
        mask = df['scenario_id'].isin([scenario1, scenario2])
        scenario_df = df[mask]

        # Check if we have enough data
        if scenario_df.shape[0] < 10:
            print(f"  Not enough data for scenarios {scenario1} vs {scenario2}. Skipping.")
            results[(scenario1, scenario2)] = {'accuracy': 0, 'error': 'Not enough data'}
            continue

        # Extract features and labels
        X = scenario_df[feature_cols].values
        # Convert scenario IDs to binary labels (0 for scenario1, 1 for scenario2)
        y = np.where(scenario_df['scenario_id'] == scenario1, 0, 1)

        print(X)
        print(y)

        # Print class distribution
        print(f"  Samples for Scenario {scenario1}: {np.sum(y == 0)}")
        print(f"  Samples for Scenario {scenario2}: {np.sum(y == 1)}")

        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25
            )

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train SVM model
            model = SVC(kernel='rbf', C=10, gamma='scale')
            # model = LogisticRegression()
            model.fit(X_train_scaled, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Store in accuracy matrix for heatmap
            accuracy_matrix[i, j] = accuracy
            accuracy_matrix[j, i] = accuracy  # Mirror for symmetric heatmap

            # Cross-validation for more robust evaluation
            cv_scores = cross_val_score(model, X, y, cv=5)

            print(f"  Test accuracy: {accuracy:.4f}")
            print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print("  Classification Report:")
            print(
                classification_report(y_test, y_pred, target_names=[f"Scenario {scenario1}", f"Scenario {scenario2}"]))

            # # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(cm)

            # Store results
            results[(scenario1, scenario2)] = {
                'accuracy': accuracy,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': cm,
                'classification_report': classification_report(y_test, y_pred,
                                                               target_names=[f"Scenario {scenario1}",
                                                                             f"Scenario {scenario2}"],
                                                               output_dict=True)
            }

        except Exception as e:
            print(f"  Error: {e}")
            results[(scenario1, scenario2)] = {'accuracy': 0, 'error': str(e)}


    # Create summary table of results
    summary_table = []
    for (s1, s2), res in results.items():
        if 'accuracy' in res and 'error' not in res:
            summary_table.append({
                'Scenario 1': s1,
                'Scenario 2': s2,
                'Accuracy': res['accuracy'],
                'CV Accuracy': res['cv_accuracy'],
                'CV Std': res['cv_std']
            })

    return results, accuracy_matrix



# Main execution
if __name__ == "__main__":
    # Load the dataset
    dataset_path = 'mfcc_dataset_with_metadata.csv'
    df = pd.read_csv(dataset_path)

    # Extract unique values from metadata columns
    metadata_columns = ['computer_name', 'scenario_id', 'room_name', 'signal_shape']
    selector_groups = []

    for column in metadata_columns:
        unique_values = sorted(df[column].unique())

        if column == 'signal_shape':
            mode = 'single'
        else:
            mode = 'multiple'

        selector_groups.append({
            'name': column,
            'items': unique_values,
            'mode': mode
        })

    # Run the filter interface
    selections = create_checkbox_interface(selector_groups)

    if selections:
        print("Filtering dataset based on your selections...")

        # Apply filters
        filtered_df = df.copy()
        for column, selected_values in selections.items():
            if selected_values:  # Only apply filter if values are selected
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]

        # Print info about filtered dataset
        print(f"Filtered dataset contains {filtered_df.shape[0]} samples")
        for column in metadata_columns:
            remaining_values = filtered_df[column].unique()
            print(f"Remaining {column} values: {sorted(remaining_values)}")

        # Run pairwise scenario classification on filtered dataset
        if len(filtered_df['scenario_id'].unique()) >= 2:
            print("\nRunning pairwise scenario classification...")
            scenario_results, accuracy_matrix = pairwise_scenario_classification(filtered_df)
        else:
            print("\nNeed at least 2 different scenarios for pairwise classification!")
            print("Please adjust your filter selections and try again.")
    else:
        print("No selections made or filter interface canceled.")

    # Create heatmap of all scenario pairs
    # Fill diagonal with 1s (perfect classification of a scenario against itself)
    np.fill_diagonal(accuracy_matrix, 1.0)

    plt.figure(figsize=(12, 10))
    scenarios = filtered_df['scenario_id'].unique()
    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=[f"S{s}" for s in scenarios],
                yticklabels=[f"S{s}" for s in scenarios])
    plt.xlabel('Scenario')
    plt.ylabel('Scenario')
    plt.title('Pairwise Scenario Classification Accuracy')
    plt.tight_layout()
    plt.savefig('scenario_pairwise_accuracy.png', dpi=300)
    plt.show()

    # summary_df = pd.DataFrame(summary_table)
    # if not summary_df.empty:
    #     summary_df = summary_df.sort_values('Accuracy', ascending=False)
    #     print("\nSummary of Pairwise Classification Results:")
    #     print(summary_df)
    #
    #     # Save to CSV
    #     summary_df.to_csv('scenario_pairwise_classification_summary.csv', index=False)
