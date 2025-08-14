#!/usr/bin/env python3
"""
GUI Visualization Panel

Handles visualization of classification results from the Classification Panel.
Supports visualization for single pair, all pairs, and group vs group classification.
"""

import os
from typing import List, Dict, Optional, Any
import streamlit as st
import pandas as pd
import numpy as np

# Try to import Plotly, fall back to basic charts if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# Try matplotlib as backup
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Session keys for classification artifacts
SK_CLASSIFICATION_RESULTS = "classification_results"
SK_LAST_MODEL_INFO = "last_model_info"
SK_CLASSIFICATION_ARTIFACTS = "classification_artifacts"


class VisualizationPanel:
    """Visualization panel implementation for classification results."""
    
    def __init__(self, scenario_manager):
        """Initialize with scenario manager for data access."""
        self.scenario_manager = scenario_manager
    
    def render(self):
        """Render the visualization panel UI."""
        st.header("Visualize - Classification Results")
        
        # Check for plotting library availability
        if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            st.error("❌ No plotting libraries available. Please install plotly or matplotlib:")
            st.code("pip install plotly")
            st.info("Or alternatively: pip install matplotlib seaborn")
            return
        
        if not PLOTLY_AVAILABLE:
            st.warning("⚠️ Plotly not available. Using basic charts with matplotlib/streamlit.")
        
        # Check for classification results
        if not self._check_results_availability():
            return
        
        # Load classification results and model info
        results = st.session_state.get(SK_CLASSIFICATION_RESULTS)
        model_info = st.session_state.get(SK_LAST_MODEL_INFO)
        
        # Display model information
        self._display_model_info(model_info)
        
        # Render visualizations based on classification mode
        mode = model_info.get('mode', 'unknown')
        
        if mode == 'single_pair':
            self._render_single_pair_visualizations(results, model_info)
        elif mode == 'all_pairs':
            self._render_all_pairs_visualizations(results, model_info)
        elif mode in ['group_vs_group', 'uploaded']:
            self._render_group_vs_group_visualizations(results, model_info)
        else:
            st.error(f"❌ Unknown classification mode: {mode}")
    
    def _check_results_availability(self) -> bool:
        """Check if classification results are available."""
        results = st.session_state.get(SK_CLASSIFICATION_RESULTS)
        model_info = st.session_state.get(SK_LAST_MODEL_INFO)
        
        if not results or not model_info:
            st.warning("⚠️ No classification results available for visualization.")
            st.info("Please run a classification in the Classification panel first.")
            
            # Show what's available
            if model_info and not results:
                st.info(f"💡 Model available ({model_info.get('model_type', 'unknown')}) but no results. Re-run classification to generate visualizations.")
            
            return False
        
        return True
    
    def _display_model_info(self, model_info: Dict[str, Any]):
        """Display information about the classification model and results."""
        st.markdown("### 🤖 Classification Summary")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.metric("Model Type", model_info.get("model_type", "Unknown").upper())
        
        with col2:
            st.metric("Feature Type", model_info.get("feature_type", "Unknown").upper())
        
        with col3:
            mode = model_info.get("mode", "Unknown").replace("_", " ").title()
            st.metric("Classification Mode", mode)
        
        with col4:
            if "test_accuracy" in model_info:
                st.metric("Test Accuracy", f"{model_info['test_accuracy']:.3f}")
            else:
                st.metric("Status", "Complete")
    
    def _render_single_pair_visualizations(self, results: Dict, model_info: Dict):
        """Render visualizations for single pair classification."""
        st.markdown("### 📊 Single Pair Classification Results")
        
        # Extract data
        labels = model_info.get('labels', ['Class A', 'Class B'])
        scenario_names = model_info.get('scenario_names', ['Scenario A', 'Scenario B'])
        
        # Performance metrics visualization
        self._render_performance_metrics(results)
        
        # Confusion matrix heatmap
        self._render_confusion_matrix_heatmap(results, labels)
        
        # Cross-validation scores
        self._render_cv_scores_chart(results)
        
        # Feature importance (if available)
        self._render_feature_importance(results, model_info)
        
        # Classification report details
        self._render_classification_report_details(results, labels, scenario_names)
    
    def _render_all_pairs_visualizations(self, results: Dict, model_info: Dict):
        """Render visualizations for all pairs classification."""
        st.markdown("### 📊 All Pairs Classification Results")
        
        # Extract data
        accuracy_matrix = results['accuracy_matrix']
        scenario_names = results['scenario_names']
        pair_results = results['pair_results']
        
        # Summary statistics
        self._render_all_pairs_summary(results)
        
        # Accuracy matrix heatmap
        self._render_accuracy_matrix_heatmap(accuracy_matrix, scenario_names)
        
        # Pair performance distribution
        self._render_pair_performance_distribution(pair_results)
        
        # Top/bottom performing pairs
        self._render_top_bottom_pairs(pair_results)
        
        # Detailed pair explorer
        self._render_pair_explorer(pair_results)
    
    def _render_group_vs_group_visualizations(self, results: Dict, model_info: Dict):
        """Render visualizations for group vs group classification."""
        st.markdown("### 📊 Group vs Group Classification Results")
        
        # Extract group names
        if model_info.get('mode') == 'group_vs_group':
            group_names = [model_info.get('group_a', 'Group A'), model_info.get('group_b', 'Group B')]
        else:
            group_names = model_info.get('label_names', ['Group A', 'Group B'])
        
        # Performance metrics visualization
        self._render_performance_metrics(results)
        
        # Confusion matrix heatmap
        self._render_confusion_matrix_heatmap(results, group_names)
        
        # Cross-validation scores
        self._render_cv_scores_chart(results)
        
        # Dataset composition (if train/test data available)
        self._render_dataset_composition(results, group_names)
        
        # Classification report details
        self._render_classification_report_details(results, group_names, group_names)
    
    def _render_performance_metrics(self, results: Dict):
        """Render performance metrics as gauges and charts."""
        st.markdown("### 🎯 Performance Metrics")
        
        # Create performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Train vs Test Accuracy
            if PLOTLY_AVAILABLE:
                fig_acc = go.Figure()
                
                fig_acc.add_trace(go.Bar(
                    x=['Training', 'Testing'],
                    y=[results['train_accuracy'], results['test_accuracy']],
                    marker_color=['lightblue', 'darkblue'],
                    text=[f"{results['train_accuracy']:.3f}", f"{results['test_accuracy']:.3f}"],
                    textposition='auto',
                ))
                
                fig_acc.update_layout(
                    title="Training vs Testing Accuracy",
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                
                st.plotly_chart(fig_acc, use_container_width=True)
            else:
                # Fallback to simple metrics and basic chart
                st.markdown("**Training vs Testing Accuracy**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Training Accuracy", f"{results['train_accuracy']:.3f}")
                with col_b:
                    st.metric("Testing Accuracy", f"{results['test_accuracy']:.3f}")
                
                # Simple bar chart using streamlit
                chart_data = pd.DataFrame({
                    'Accuracy': [results['train_accuracy'], results['test_accuracy']],
                    'Type': ['Training', 'Testing']
                })
                st.bar_chart(chart_data.set_index('Type'))
        
        with col2:
            # Cross-validation metrics
            cv_mean = results['cv_mean']
            cv_std = results['cv_std']
            
            if PLOTLY_AVAILABLE:
                # Cross-validation gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = cv_mean,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"CV Score ± {cv_std:.3f}"},
                    delta = {'reference': 0.5},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                # Fallback to simple metrics
                st.markdown("**Cross-Validation Score**")
                st.metric("CV Mean", f"{cv_mean:.3f}")
                st.metric("CV Std Dev", f"{cv_std:.3f}")
                
                # Progress bar visualization
                st.markdown("**Performance Level:**")
                progress_val = min(cv_mean, 1.0)
                st.progress(progress_val)
                
                if cv_mean >= 0.9:
                    st.success("Excellent performance!")
                elif cv_mean >= 0.8:
                    st.success("Good performance")
                elif cv_mean >= 0.7:
                    st.warning("Moderate performance")
                else:
                    st.error("Low performance")
    
    def _render_confusion_matrix_heatmap(self, results: Dict, labels: List[str]):
        """Render confusion matrix as an interactive heatmap."""
        st.markdown("### 🔀 Confusion Matrix")
        
        cm = results['confusion_matrix']
        
        if PLOTLY_AVAILABLE:
            # Create Plotly heatmap
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=labels,
                y=labels,
                color_continuous_scale='Blues',
                aspect="auto"
            )
            
            # Add text annotations
            for i in range(len(labels)):
                for j in range(len(labels)):
                    fig.add_annotation(
                        x=j, y=i,
                        text=str(cm[i][j]),
                        showarrow=False,
                        font=dict(color="white" if cm[i][j] > cm.max()/2 else "black", size=16)
                    )
            
            fig.update_layout(
                title="Confusion Matrix Heatmap",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Fallback to DataFrame display
            st.markdown("**Confusion Matrix:**")
            cm_df = pd.DataFrame(cm, index=[f"True {label}" for label in labels], 
                               columns=[f"Pred {label}" for label in labels])
            st.dataframe(cm_df, use_container_width=True)
            
            # Add simple visualization using matplotlib if available
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
        
        # Add interpretation
        total_samples = cm.sum()
        correct_predictions = np.trace(cm)
        
        st.caption(f"""
        **Matrix Interpretation:**
        - Total test samples: {total_samples}
        - Correct predictions: {correct_predictions} ({correct_predictions/total_samples*100:.1f}%)
        - Diagonal values represent correct classifications
        - Off-diagonal values represent misclassifications
        """)
    
    def _render_cv_scores_chart(self, results: Dict):
        """Render cross-validation scores as a chart."""
        st.markdown("### 📈 Cross-Validation Performance")
        
        cv_scores = results['cv_scores']
        folds = list(range(1, len(cv_scores) + 1))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if PLOTLY_AVAILABLE:
                # Line chart of CV scores
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=folds,
                    y=cv_scores,
                    mode='lines+markers',
                    name='CV Score',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                # Add mean line
                mean_score = np.mean(cv_scores)
                fig.add_hline(
                    y=mean_score,
                    line_dash="dash",
                    annotation_text=f"Mean: {mean_score:.3f}",
                    annotation_position="bottom right"
                )
                
                fig.update_layout(
                    title="Cross-Validation Scores by Fold",
                    xaxis_title="Fold",
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to streamlit chart
                st.markdown("**Cross-Validation Scores by Fold:**")
                cv_data = pd.DataFrame({
                    'Fold': folds,
                    'CV Score': cv_scores
                })
                st.line_chart(cv_data.set_index('Fold'))
                
                # Show mean line info
                mean_score = np.mean(cv_scores)
                st.info(f"Mean CV Score: {mean_score:.3f}")
        
        with col2:
            # Statistics table
            st.markdown("**CV Statistics:**")
            cv_stats = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    f"{np.mean(cv_scores):.3f}",
                    f"{np.std(cv_scores):.3f}",
                    f"{np.min(cv_scores):.3f}",
                    f"{np.max(cv_scores):.3f}",
                    f"{np.max(cv_scores) - np.min(cv_scores):.3f}"
                ]
            })
            st.dataframe(cv_stats, hide_index=True)
            
            # Fold details
            with st.expander("Fold Details"):
                for i, score in enumerate(cv_scores):
                    st.text(f"Fold {i+1}: {score:.3f}")
    
    def _render_accuracy_matrix_heatmap(self, accuracy_matrix: np.ndarray, scenario_names: List[str]):
        """Render accuracy matrix for all pairs classification."""
        st.markdown("### 🎯 Pairwise Accuracy Matrix")
        
        # Create symmetric matrix for display
        display_matrix = accuracy_matrix.copy()
        np.fill_diagonal(display_matrix, 1.0)  # Perfect accuracy on diagonal
        
        # Create heatmap
        fig = px.imshow(
            display_matrix,
            labels=dict(x="Scenario", y="Scenario", color="Accuracy"),
            x=scenario_names,
            y=scenario_names,
            color_continuous_scale='RdYlGn',
            zmin=0, zmax=1,
            aspect="auto"
        )
        
        # Add text annotations for non-diagonal elements
        for i in range(len(scenario_names)):
            for j in range(len(scenario_names)):
                if i != j and accuracy_matrix[i][j] > 0:
                    fig.add_annotation(
                        x=j, y=i,
                        text=f"{accuracy_matrix[i][j]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if accuracy_matrix[i][j] < 0.5 else "black", size=10)
                    )
        
        fig.update_layout(
            title="Pairwise Classification Accuracy",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_pair_performance_distribution(self, pair_results: Dict):
        """Render distribution of pair performance."""
        st.markdown("### 📊 Pair Performance Distribution")
        
        accuracies = [result['accuracy'] for result in pair_results.values()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                x=accuracies,
                bins=20,
                title="Distribution of Pairwise Accuracies",
                labels={'x': 'Accuracy', 'y': 'Number of Pairs'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                y=accuracies,
                title="Accuracy Distribution Summary",
                labels={'y': 'Accuracy'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_all_pairs_summary(self, results: Dict):
        """Render summary statistics for all pairs classification."""
        accuracy_matrix = results['accuracy_matrix']
        non_zero_accuracies = accuracy_matrix[accuracy_matrix > 0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pairs", results['total_pairs'])
        
        with col2:
            st.metric("Mean Accuracy", f"{np.mean(non_zero_accuracies):.3f}")
        
        with col3:
            st.metric("Best Accuracy", f"{np.max(non_zero_accuracies):.3f}")
        
        with col4:
            st.metric("Worst Accuracy", f"{np.min(non_zero_accuracies):.3f}")
    
    def _render_top_bottom_pairs(self, pair_results: Dict):
        """Render top and bottom performing pairs."""
        st.markdown("### 🏆 Best and Worst Performing Pairs")
        
        # Sort pairs by accuracy
        sorted_pairs = sorted(pair_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🥇 Top 5 Pairs:**")
            for pair_name, result in sorted_pairs[:5]:
                pair_display = pair_name.replace('_vs_', ' vs ')
                st.metric(pair_display, f"{result['accuracy']:.3f}")
        
        with col2:
            st.markdown("**🥉 Bottom 5 Pairs:**")
            for pair_name, result in sorted_pairs[-5:]:
                pair_display = pair_name.replace('_vs_', ' vs ')
                st.metric(pair_display, f"{result['accuracy']:.3f}")
    
    def _render_pair_explorer(self, pair_results: Dict):
        """Render detailed pair explorer."""
        st.markdown("### 🔍 Detailed Pair Analysis")
        
        # Select pair to explore
        pair_names = list(pair_results.keys())
        pair_display_names = [name.replace('_vs_', ' vs ') for name in pair_names]
        
        selected_display = st.selectbox(
            "Select pair to analyze:",
            options=pair_display_names,
            help="Choose a specific pair for detailed analysis"
        )
        
        if selected_display:
            # Find the original key
            selected_pair = None
            for i, display_name in enumerate(pair_display_names):
                if display_name == selected_display:
                    selected_pair = pair_names[i]
                    break
            
            if selected_pair and selected_pair in pair_results:
                result = pair_results[selected_pair]
                
                # Display pair details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{result['accuracy']:.3f}")
                
                with col2:
                    st.metric("CV Mean", f"{result['cv_mean']:.3f}")
                
                with col3:
                    st.metric("CV Std", f"{result['cv_std']:.3f}")
                
                # Confusion matrix for this pair
                if 'confusion_matrix' in result:
                    st.markdown(f"**Confusion Matrix for {selected_display}:**")
                    cm_df = pd.DataFrame(
                        result['confusion_matrix'],
                        index=result['scenarios'],
                        columns=result['scenarios']
                    )
                    st.dataframe(cm_df, use_container_width=True)
    
    def _render_dataset_composition(self, results: Dict, group_names: List[str]):
        """Render dataset composition visualization."""
        X_train = results.get('X_train')
        y_train = results.get('y_train')
        X_test = results.get('X_test')
        y_test = results.get('y_test')
        
        if X_train is not None and y_train is not None:
            st.markdown("### 📊 Dataset Composition")
            
            # Calculate group distributions
            train_counts = [sum(1 for y in y_train if y == i) for i in range(len(group_names))]
            test_counts = [sum(1 for y in y_test if y == i) for i in range(len(group_names))]
            
            # Create stacked bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Training',
                x=group_names,
                y=train_counts,
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Testing',
                x=group_names,
                y=test_counts,
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title='Dataset Composition by Group',
                xaxis_title='Groups',
                yaxis_title='Number of Samples',
                barmode='stack',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_feature_importance(self, results: Dict, model_info: Dict):
        """Render feature importance if available."""
        # Note: Feature importance would need to be calculated during classification
        # This is a placeholder for future implementation
        if 'feature_importance' in results:
            st.markdown("### 🎯 Feature Importance")
            
            importance = results['feature_importance']
            feature_names = model_info.get('feature_names', [f'Feature_{i}' for i in range(len(importance))])
            
            if PLOTLY_AVAILABLE:
                # Create bar chart
                fig = px.bar(
                    x=importance,
                    y=feature_names,
                    orientation='h',
                    title="Feature Importance",
                    labels={'x': 'Importance', 'y': 'Features'}
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(importance_df, hide_index=True)
                
                # Simple bar chart
                st.bar_chart(importance_df.set_index('Feature')['Importance'])
        else:
            with st.expander("🎯 Feature Importance (Not Available)"):
                st.info("Feature importance analysis is not available for this classification. This would require additional computation during the classification phase.")
    
    def _render_classification_report_details(self, results: Dict, labels: List[str], scenario_names: List[str]):
        """Render detailed classification report."""
        st.markdown("### 📋 Detailed Classification Report")
        
        with st.expander("Full Classification Report"):
            st.text(results['classification_report'])
        
        # Summary information
        st.markdown("**Classification Summary:**")
        
        summary_info = f"""
        - **Scenarios/Groups**: {', '.join(scenario_names)}
        - **Labels**: {', '.join(labels)}
        - **Model Performance**: {results['test_accuracy']:.1%} test accuracy
        - **Cross-Validation**: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}
        - **Training Samples**: {results.get('train_samples', 'N/A')}
        - **Testing Samples**: {results.get('test_samples', 'N/A')}
        """
        
        st.markdown(summary_info)


# Simple fallback visualization functions for when Plotly is not available
def create_simple_bar_chart(data: Dict, title: str):
    """Create a simple bar chart using streamlit."""
    df = pd.DataFrame(list(data.items()), columns=['Category', 'Value'])
    st.subheader(title)
    st.bar_chart(df.set_index('Category'))


def create_simple_line_chart(x_data: List, y_data: List, title: str):
    """Create a simple line chart using streamlit."""
    df = pd.DataFrame({'X': x_data, 'Y': y_data})
    st.subheader(title)
    st.line_chart(df.set_index('X'))