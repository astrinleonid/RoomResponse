#!/usr/bin/env python3
"""
Scenario Classifier

Simple ML model to discriminate between two scenario subfolders using their
spectrum or MFCC features.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ScenarioClassifier:
    """ML classifier to discriminate between two audio scenarios."""

    def __init__(self, model_type='svm', feature_type='spectrum'):
        """
        Initialize the classifier.

        Args:
            model_type (str): Type of model - 'svm' or 'logistic'
            feature_type (str): Type of features - 'spectrum' or 'mfcc'
        """
        self.model_type = model_type
        self.feature_type = feature_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None

        # Initialize model
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42, probability=True)
        elif model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError("model_type must be 'svm' or 'logistic'")

    def _get_feature_filename(self):
        """Get the appropriate CSV filename based on feature type."""
        if self.feature_type == 'spectrum':
            return 'spectrum.csv'
        elif self.feature_type == 'mfcc':
            return 'features.csv'
        else:
            raise ValueError("feature_type must be 'spectrum' or 'mfcc'")

    def load_scenario_data(self, scenario_folder, label):
        """
        Load feature data from a scenario folder.

        Args:
            scenario_folder (str): Path to scenario folder
            label (str): Label for this scenario

        Returns:
            pd.DataFrame: DataFrame with features and labels
        """
        feature_file = os.path.join(scenario_folder, self._get_feature_filename())

        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found: {feature_file}")

        # Load features
        df = pd.read_csv(feature_file)

        # Add label column
        df['scenario_label'] = label
        df['scenario_folder'] = os.path.basename(scenario_folder)

        print(f"Loaded {len(df)} samples from {os.path.basename(scenario_folder)}")

        return df

    def prepare_dataset(self, scenario1_folder, scenario2_folder,
                        scenario1_label=None, scenario2_label=None):
        """
        Prepare dataset by combining two scenario folders.

        Args:
            scenario1_folder (str): Path to first scenario folder
            scenario2_folder (str): Path to second scenario folder
            scenario1_label (str): Label for first scenario (default: folder name)
            scenario2_label (str): Label for second scenario (default: folder name)

        Returns:
            tuple: (X, y, feature_names, label_names)
        """
        # Use folder names as labels if not provided
        if scenario1_label is None:
            scenario1_label = os.path.basename(scenario1_folder)
        if scenario2_label is None:
            scenario2_label = os.path.basename(scenario2_folder)

        # Load data from both scenarios
        df1 = self.load_scenario_data(scenario1_folder, scenario1_label)
        df2 = self.load_scenario_data(scenario2_folder, scenario2_label)

        # Combine datasets
        combined_df = pd.concat([df1, df2], ignore_index=True)

        # Separate features and labels
        feature_cols = [col for col in combined_df.columns
                        if col.startswith('freq_' if self.feature_type == 'spectrum' else 'mfcc_')]

        if not feature_cols:
            raise ValueError(f"No {self.feature_type} features found in CSV files")

        X = combined_df[feature_cols].values
        y = combined_df['scenario_label'].values

        # Store feature names for later analysis
        self.feature_names = feature_cols

        print(f"\nDataset prepared:")
        print(f"Feature type: {self.feature_type}")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]}")
        print(f"Classes: {np.unique(y)}")
        print(f"Class distribution:")
        for label in np.unique(y):
            count = np.sum(y == label)
            print(f"  {label}: {count} samples ({count / len(y) * 100:.1f}%)")

        return X, y, feature_cols, [scenario1_label, scenario2_label]

    def train_and_evaluate(self, X, y, test_size=0.3, cv_folds=5):
        """
        Train the model and evaluate performance.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            test_size (float): Fraction of data for testing
            cv_folds (int): Number of cross-validation folds

        Returns:
            dict: Evaluation results
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print(f"\nTraining {self.model_type.upper()} model...")
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        # Calculate scores
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='accuracy'
        )

        # Get class probabilities for ROC analysis
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_proba': train_proba,
            'test_proba': test_proba,
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'classification_report': classification_report(
                y_test, test_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
        }

        return results

    def print_results(self, results):
        """Print evaluation results."""
        print(f"\n{'=' * 50}")
        print(f"MODEL EVALUATION RESULTS")
        print(f"{'=' * 50}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Features: {self.feature_type.upper()}")
        print(f"\nPerformance Metrics:")
        print(f"Training Accuracy: {results['train_accuracy']:.3f}")
        print(f"Test Accuracy: {results['test_accuracy']:.3f}")
        print(f"Cross-Validation: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")

        print(f"\nDetailed Classification Report:")
        print("-" * 40)
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"{class_name:15} - Precision: {metrics['precision']:.3f}, "
                      f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")

        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])

    def plot_results(self, results, save_path=None):
        """
        Create visualization plots for the results.

        Args:
            results (dict): Evaluation results
            save_path (str): Optional path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_type.upper()} Classification Results - {self.feature_type.upper()} Features',
                     fontsize=16, fontweight='bold')

        # 1. Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # 2. Cross-Validation Scores
        axes[0, 1].bar(range(len(results['cv_scores'])), results['cv_scores'],
                       color='skyblue', alpha=0.7)
        axes[0, 1].axhline(y=results['cv_mean'], color='red', linestyle='--',
                           label=f'Mean: {results["cv_mean"]:.3f}')
        axes[0, 1].set_title('Cross-Validation Scores')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)

        # 3. Feature Importance Analysis
        if self.model_type == 'logistic' and hasattr(self.model, 'coef_'):
            # For logistic regression: use coefficients
            feature_importance = np.abs(self.model.coef_[0])
            importance_title = 'Feature Importance (|Coefficients|)'
            importance_xlabel = 'Absolute Coefficient Value'
        elif self.model_type == 'svm' and hasattr(self.model, 'dual_coef_'):
            # For SVM: estimate feature importance using permutation-like method
            # Use the magnitude of the decision function gradient as proxy
            try:
                # Get support vectors and their coefficients
                support_vectors = self.model.support_vectors_
                dual_coef = self.model.dual_coef_[0]

                # Calculate feature importance as weighted sum of support vector features
                feature_importance = np.abs(np.sum(dual_coef[:, np.newaxis] * support_vectors, axis=0))
                importance_title = 'Feature Importance (SVM Support Vector Weights)'
                importance_xlabel = 'Importance Score'
            except Exception:
                # Fallback: use variance-based importance
                X_combined = np.vstack([results['X_train'], results['X_test']])
                y_combined = np.hstack([results['y_train'], results['y_test']])

                # Calculate class-wise feature variance differences
                class0_data = X_combined[y_combined == 0]
                class1_data = X_combined[y_combined == 1]

                mean_diff = np.abs(np.mean(class1_data, axis=0) - np.mean(class0_data, axis=0))
                var_ratio = np.var(class1_data, axis=0) / (np.var(class0_data, axis=0) + 1e-8)

                feature_importance = mean_diff * np.log(var_ratio + 1e-8)
                feature_importance = np.abs(feature_importance)

                importance_title = 'Feature Importance (Class Difference Analysis)'
                importance_xlabel = 'Discriminative Score'
        else:
            # Fallback: variance-based importance
            X_combined = np.vstack([results['X_train'], results['X_test']])
            y_combined = np.hstack([results['y_train'], results['y_test']])

            class0_data = X_combined[y_combined == 0]
            class1_data = X_combined[y_combined == 1]

            mean_diff = np.abs(np.mean(class1_data, axis=0) - np.mean(class0_data, axis=0))
            feature_importance = mean_diff

            importance_title = 'Feature Importance (Mean Difference)'
            importance_xlabel = 'Absolute Mean Difference'

        # Plot top features
        num_features_to_show = min(30, len(feature_importance))
        top_features_idx = np.argsort(feature_importance)[-num_features_to_show:]

        # For spectrum features, convert indices to frequencies
        if self.feature_type == 'spectrum' and hasattr(self, 'feature_names'):
            # Estimate sample rate and calculate frequencies
            sample_rate = 16000  # Default assumption
            n_fft_bins = len(feature_importance)
            frequencies = np.linspace(0, sample_rate / 2, n_fft_bins)

            feature_labels = [f'{frequencies[i]:.0f} Hz' for i in top_features_idx]
        else:
            # Use feature names or indices
            if hasattr(self, 'feature_names') and self.feature_names:
                feature_labels = [self.feature_names[i] for i in top_features_idx]
            else:
                feature_labels = [f'Feature {i}' for i in top_features_idx]

        axes[1, 0].barh(range(len(top_features_idx)), feature_importance[top_features_idx],
                        color='skyblue', alpha=0.7)
        axes[1, 0].set_yticks(range(len(top_features_idx)))
        axes[1, 0].set_yticklabels(feature_labels, fontsize=8)
        axes[1, 0].set_title(importance_title)
        axes[1, 0].set_xlabel(importance_xlabel)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Feature Importance (for logistic regression) or Performance Summary
        if self.model_type == 'logistic' and hasattr(self.model, 'coef_'):
            feature_importance = np.abs(self.model.coef_[0])
            top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features

            axes[1, 1].barh(range(len(top_features_idx)), feature_importance[top_features_idx])
            axes[1, 1].set_yticks(range(len(top_features_idx)))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in top_features_idx])
            axes[1, 1].set_title('Top 20 Feature Importance (|Coefficients|)')
            axes[1, 1].set_xlabel('Absolute Coefficient Value')
        else:
            # Performance summary for SVM
            metrics = ['Train Acc', 'Test Acc', 'CV Mean']
            values = [results['train_accuracy'], results['test_accuracy'], results['cv_mean']]
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]

            bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
            axes[1, 1].set_title('Performance Summary')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_ylim(0, 1)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def save_model_info(self, results, scenario1_folder, scenario2_folder, output_dir):
        """
        Save model information and results to files.

        Args:
            results (dict): Evaluation results
            scenario1_folder (str): First scenario folder path
            scenario2_folder (str): Second scenario folder path
            output_dir (str): Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        results_text = f"""
Scenario Classification Results
{'=' * 40}

Scenarios Compared:
- Scenario 1: {os.path.basename(scenario1_folder)}
- Scenario 2: {os.path.basename(scenario2_folder)}

Model Configuration:
- Algorithm: {self.model_type.upper()}
- Features: {self.feature_type.upper()}
- Feature Dimensions: {len(self.feature_names)}

Performance Metrics:
- Training Accuracy: {results['train_accuracy']:.4f}
- Test Accuracy: {results['test_accuracy']:.4f}
- Cross-Validation Mean: {results['cv_mean']:.4f}
- Cross-Validation Std: {results['cv_std']:.4f}

Cross-Validation Scores:
{', '.join([f'{score:.3f}' for score in results['cv_scores']])}

Confusion Matrix:
{results['confusion_matrix']}

Classification Report:
"""

        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                results_text += f"\n{class_name}:\n"
                results_text += f"  Precision: {metrics['precision']:.4f}\n"
                results_text += f"  Recall: {metrics['recall']:.4f}\n"
                results_text += f"  F1-Score: {metrics['f1-score']:.4f}\n"

        # Save results
        with open(os.path.join(output_dir, 'classification_results.txt'), 'w') as f:
            f.write(results_text)

        # Save feature importance (if available)
        if self.model_type == 'logistic' and hasattr(self.model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_[0]),
                'coefficient': self.model.coef_[0]
            }).sort_values('importance', ascending=False)

            feature_importance.to_csv(
                os.path.join(output_dir, 'feature_importance.csv'),
                index=False
            )

        print(f"Results saved to: {output_dir}")


def main():
    """Main entry point for scenario classification."""
    parser = argparse.ArgumentParser(
        description='Train ML model to discriminate between two audio scenarios'
    )

    # Required arguments
    parser.add_argument('scenario1', help='Path to first scenario folder')
    parser.add_argument('scenario2', help='Path to second scenario folder')

    # Optional arguments
    parser.add_argument('--model', choices=['svm', 'logistic'], default='svm',
                        help='ML model type (default: svm)')
    parser.add_argument('--features', choices=['spectrum', 'mfcc'], default='spectrum',
                        help='Feature type to use (default: spectrum)')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Fraction of data for testing (default: 0.3)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--output-dir', default='classification_results',
                        help='Directory to save results (default: classification_results)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--scenario1-label', help='Custom label for scenario 1')
    parser.add_argument('--scenario2-label', help='Custom label for scenario 2')

    args = parser.parse_args()

    # Validate input folders
    if not os.path.exists(args.scenario1):
        print(f"Error: Scenario 1 folder not found: {args.scenario1}")
        return

    if not os.path.exists(args.scenario2):
        print(f"Error: Scenario 2 folder not found: {args.scenario2}")
        return

    print(f"Scenario Classification")
    print(f"{'=' * 30}")
    print(f"Scenario 1: {args.scenario1}")
    print(f"Scenario 2: {args.scenario2}")
    print(f"Model: {args.model.upper()}")
    print(f"Features: {args.features.upper()}")

    try:
        # Initialize classifier
        classifier = ScenarioClassifier(
            model_type=args.model,
            feature_type=args.features
        )

        # Prepare dataset
        X, y, feature_names, label_names = classifier.prepare_dataset(
            args.scenario1, args.scenario2,
            args.scenario1_label, args.scenario2_label
        )

        # Train and evaluate
        results = classifier.train_and_evaluate(
            X, y, test_size=args.test_size, cv_folds=args.cv_folds
        )

        # Print results
        classifier.print_results(results)

        # Generate plots
        if not args.no_plot:
            plot_path = os.path.join(args.output_dir, 'classification_plots.png')
            classifier.plot_results(results, save_path=plot_path)

        # Save results
        classifier.save_model_info(results, args.scenario1, args.scenario2, args.output_dir)

        # Final summary
        print(f"\n{'=' * 50}")
        print(f"CLASSIFICATION COMPLETE")
        print(f"{'=' * 50}")
        print(f"Test Accuracy: {results['test_accuracy']:.3f}")
        print(f"Cross-Validation: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")

        if results['test_accuracy'] > 0.9:
            print("🎉 Excellent classification performance!")
        elif results['test_accuracy'] > 0.8:
            print("✅ Good classification performance!")
        elif results['test_accuracy'] > 0.7:
            print("⚠️  Moderate classification performance")
        else:
            print("❌ Poor classification performance - scenarios may be too similar")

    except Exception as e:
        print(f"Error during classification: {e}")
        raise


if __name__ == "__main__":
    main()