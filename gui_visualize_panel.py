#!/usr/bin/env python3
"""
GUI Visualization Panel (updated)

- All charts collapsed by default (expanders default to expanded=False)
- Uses concise scenario labels (scenario number when possible) in all charts
- Fix: Plotly histogram uses `nbins` (not `bins`)
- Adds Feature Importance to Group-vs-Group (and Single Pair if available)
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

    # ---------------------- helpers ----------------------
    def _short_name(self, full: str) -> str:
        """
        Return a compact scenario label.
        Priority: parsed number (e.g., '1.81') -> folder base name -> original.
        """
        if not isinstance(full, str) or not full:
            return str(full)
        base = os.path.basename(full)

        # Use ScenarioManager parser if available
        try:
            if self.scenario_manager and hasattr(self.scenario_manager, "parse_scenario_folder_name"):
                num, comp, room = self.scenario_manager.parse_scenario_folder_name(base)
                if str(num).strip():
                    return str(num).strip()
        except Exception:
            pass

        # Fallback: extract number after '-Scenario'
        # Examples:
        #  "PC-Scenario1.81-RoomX" -> "1.81"
        #  "Scenario5.2" -> "5.2"
        import re
        m = re.search(r"(?i)scenario([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)", base)
        if m:
            return m.group(1)

        # If nothing worked, prefer base name to avoid long paths
        return base

    def _shorten_many(self, names: List[str]) -> List[str]:
        return [self._short_name(x) for x in names]

    # ---------------------- entry ----------------------
    def render(self):
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
        model_info = st.session_state.get(SK_LAST_MODEL_INFO) or {}
        mode = model_info.get('mode', 'unknown')

        # Display model information
        self._display_model_info(model_info)

        # Render visualizations based on classification mode
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
            return False
        return True

    # ---------------------- summary header ----------------------
    def _display_model_info(self, model_info: Dict[str, Any]):
        st.markdown("### 🤖 Classification Summary")

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            st.metric("Model Type", str(model_info.get("model_type", "Unknown")).upper())

        with col2:
            st.metric("Feature Type", str(model_info.get("feature_type", "Unknown")).upper())

        with col3:
            mode = str(model_info.get("mode", "Unknown")).replace("_", " ").title()
            st.metric("Classification Mode", mode)

        with col4:
            met = model_info.get("metrics", {})
            if "test_accuracy" in met:
                st.metric("Test Accuracy", f"{float(met['test_accuracy']):.3f}")
            else:
                st.metric("Status", "Complete")

    # ---------------------- single pair ----------------------
    def _render_single_pair_visualizations(self, results: Dict, model_info: Dict):
        st.markdown("### 📊 Single Pair Classification Results")

        # Extract data
        labels = model_info.get('labels') or results.get('label_names') or ['Class A', 'Class B']
        scenario_names = model_info.get('scenarios') or model_info.get('scenario_names') or labels
        short_names = self._shorten_many(scenario_names)

        # Performance metrics visualization
        self._render_performance_metrics(results)

        # Confusion matrix heatmap
        self._render_confusion_matrix_heatmap(results, labels)

        # Cross-validation scores
        self._render_cv_scores_chart(results)

        # Feature importance (if available)
        self._render_feature_importance(results, model_info)

        # Classification report details
        self._render_classification_report_details(results, labels, short_names)

    # ---------------------- all pairs ----------------------
    def _render_all_pairs_visualizations(self, results: Dict, model_info: Dict):
        st.markdown("### 📊 All Pairs Classification Results")

        # Extract data safely
        accuracy_matrix = results.get('accuracy_matrix')
        scenario_names = results.get('scenario_names', [])
        short_names = self._shorten_many(scenario_names)
        pair_results = results.get('pair_results', {})
        if accuracy_matrix is None or len(scenario_names) == 0:
            st.error("No pairwise grid cached. Please re-run All Pairs in the Classify panel.")
            return

        # Summary statistics
        self._render_all_pairs_summary(results)

        # Accuracy matrix heatmap
        self._render_accuracy_matrix_heatmap(accuracy_matrix, short_names)

        # Pair performance distribution
        self._render_pair_performance_distribution(pair_results)

        # Top/bottom performing pairs
        self._render_top_bottom_pairs(pair_results)

        # Detailed pair explorer
        self._render_pair_explorer(pair_results)

    # ---------------------- group vs group ----------------------
    def _render_group_vs_group_visualizations(self, results: Dict, model_info: Dict):
        st.markdown("### 📊 Group vs Group Classification Results")

        # Extract group names
        if model_info.get('mode') == 'group_vs_group':
            group_names = [model_info.get('labels', ['Group A', 'Group B'])[0] if model_info.get('labels') else model_info.get('group_a', 'Group A'),
                           model_info.get('labels', ['Group A', 'Group B'])[1] if model_info.get('labels') else model_info.get('group_b', 'Group B')]
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

        # Feature importance (NEW)
        self._render_feature_importance(results, model_info)

        # Classification report details
        self._render_classification_report_details(results, group_names, group_names)

    # ---------------------- shared blocks ----------------------
    def _render_performance_metrics(self, results: Dict):
        """Render performance metrics."""
        with st.expander("🎯 Performance Metrics", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                if PLOTLY_AVAILABLE:
                    train = float(results.get('train_accuracy', 0.0))
                    test = float(results.get('test_accuracy', 0.0))
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Bar(
                        x=['Training', 'Testing'],
                        y=[train, test],
                        text=[f"{train:.3f}", f"{test:.3f}"],
                        textposition='auto',
                    ))
                    fig_acc.update_layout(
                        title="Training vs Testing Accuracy",
                        yaxis_title="Accuracy",
                        yaxis=dict(range=[0, 1]),
                        height=380
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                else:
                    st.markdown("**Training vs Testing Accuracy**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Training Accuracy", f"{results.get('train_accuracy', 0):.3f}")
                    with col_b:
                        st.metric("Testing Accuracy", f"{results.get('test_accuracy', 0):.3f}")

            with col2:
                cv_mean = float(results.get('cv_mean', 0.0))
                cv_std = float(results.get('cv_std', 0.0))
                if PLOTLY_AVAILABLE:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=cv_mean,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"CV Mean (±{cv_std:.3f})"},
                        gauge={
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "yellow"},
                                {'range': [0.8, 1], 'color': "green"},
                            ],
                        }
                    ))
                    fig_gauge.update_layout(height=380)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                else:
                    st.metric("CV Mean", f"{cv_mean:.3f}")
                    st.metric("CV Std Dev", f"{cv_std:.3f}")

    def _render_confusion_matrix_heatmap(self, results: Dict, labels: List[str]):
        """Render confusion matrix as an interactive heatmap."""
        with st.expander("🔀 Confusion Matrix", expanded=False):
            cm = np.asarray(results.get('confusion_matrix', []))
            if cm.size == 0:
                st.info("Confusion matrix not available.")
                return

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
                maxv = cm.max() if cm.size else 0
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        val = int(cm[i][j])
                        fig.add_annotation(
                            x=j, y=i,
                            text=str(val),
                            showarrow=False,
                            font=dict(color="white" if (maxv and val > maxv/2) else "black", size=14)
                        )

                fig.update_layout(
                    title="Confusion Matrix Heatmap",
                    height=420
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("**Confusion Matrix:**")
                cm_df = pd.DataFrame(cm, index=[f"True {label}" for label in labels],
                                     columns=[f"Pred {label}" for label in labels])
                st.dataframe(cm_df, use_container_width=True)

                if MATPLOTLIB_AVAILABLE:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=labels, yticklabels=labels, ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)

            total_samples = int(cm.sum())
            correct_predictions = int(np.trace(cm))
            if total_samples > 0:
                st.caption(
                    f"**Total:** {total_samples} | **Correct:** {correct_predictions} "
                    f"({correct_predictions / total_samples * 100:.1f}%)"
                )

    def _render_cv_scores_chart(self, results: Dict):
        """Render cross-validation scores as a chart."""
        with st.expander("📈 Cross-Validation Performance", expanded=False):
            cv_scores = np.asarray(results.get('cv_scores', []), dtype=float)
            if cv_scores.size == 0:
                st.caption("No CV fold scores available.")
                return
            folds = list(range(1, len(cv_scores) + 1))

            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=folds, y=cv_scores.tolist(),
                    mode='lines+markers',
                    name='CV Score'
                ))
                mean_score = float(np.mean(cv_scores))
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
                    height=420
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                cv_data = pd.DataFrame({'Fold': folds, 'CV Score': cv_scores})
                st.line_chart(cv_data.set_index('Fold'))
                st.info(f"Mean CV Score: {float(np.mean(cv_scores)):.3f}")

    def _render_accuracy_matrix_heatmap(self, accuracy_matrix: np.ndarray, scenario_names: List[str]):
        """Render accuracy matrix for all pairs classification."""
        with st.expander("🎯 Pairwise Accuracy Matrix", expanded=False):
            if not PLOTLY_AVAILABLE:
                st.warning("Plotly required for interactive heatmap.")
                st.dataframe(pd.DataFrame(accuracy_matrix, index=scenario_names, columns=scenario_names))
                return

            # Create symmetric display with perfect diagonal
            display_matrix = np.array(accuracy_matrix, dtype=float)
            if display_matrix.shape[0] == display_matrix.shape[1]:
                np.fill_diagonal(display_matrix, 1.0)

            fig = px.imshow(
                display_matrix,
                labels=dict(x="Scenario", y="Scenario", color="Accuracy"),
                x=scenario_names,
                y=scenario_names,
                color_continuous_scale='RdYlGn',
                zmin=0, zmax=1,
                aspect="auto"
            )

            n = len(scenario_names)
            for i in range(n):
                for j in range(n):
                    if i != j and not np.isnan(accuracy_matrix[i][j]) and accuracy_matrix[i][j] > 0:
                        fig.add_annotation(
                            x=j, y=i,
                            text=f"{accuracy_matrix[i][j]:.2f}",
                            showarrow=False,
                            font=dict(size=10)
                        )

            fig.update_layout(
                title="Pairwise Classification Accuracy",
                height=640
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_pair_performance_distribution(self, pair_results: Dict):
        """Render distribution of pair performance."""
        with st.expander("📊 Pair Performance Distribution", expanded=False):
            if not pair_results:
                st.caption("No per-pair results available.")
                return

            accuracies = [float(v.get('accuracy', 0.0)) for v in pair_results.values()]

            if PLOTLY_AVAILABLE:
                # Plotly express wants `nbins`, not `bins`
                try:
                    fig = px.histogram(
                        x=accuracies,
                        nbins=20,
                        title="Distribution of Pairwise Accuracies",
                        labels={'x': 'Accuracy', 'y': 'Number of Pairs'}
                    )
                except TypeError:
                    # Very old plotly fallback via graph_objects
                    fig = go.Figure(data=[go.Histogram(x=accuracies, nbinsx=20)])
                    fig.update_layout(title="Distribution of Pairwise Accuracies",
                                      xaxis_title="Accuracy", yaxis_title="Number of Pairs")
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.box(
                    y=accuracies,
                    title="Accuracy Distribution Summary",
                    labels={'y': 'Accuracy'}
                )
                fig2.update_layout(height=380)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.bar_chart(pd.Series(accuracies, name="Accuracy"))
                st.caption("Box plot unavailable without Plotly.")

    def _render_all_pairs_summary(self, results: Dict):
        """Render summary statistics for all pairs classification."""
        with st.expander("ℹ️ All Pairs Summary", expanded=False):
            am = np.asarray(results.get('accuracy_matrix', []), dtype=float)
            if am.size == 0:
                st.caption("No grid values.")
                return
            tri = np.triu_indices_from(am, k=1)
            vals = am[tri] if tri[0].size else am.flatten()
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                st.caption("No valid pair accuracies.")
                return
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pairs", int(results.get('summary', {}).get('total_pairs', len(vals))))
            with col2:
                st.metric("Mean Accuracy", f"{float(np.mean(vals)):.3f}")
            with col3:
                st.metric("Best Accuracy", f"{float(np.max(vals)):.3f}")
            with col4:
                st.metric("Worst Accuracy", f"{float(np.min(vals)):.3f}")

    def _render_top_bottom_pairs(self, pair_results: Dict):
        """Render top and bottom performing pairs."""
        with st.expander("🏆 Best & Worst Pairs", expanded=False):
            if not pair_results:
                st.caption("No per-pair results available.")
                return

            # Sort pairs by accuracy
            items = []
            for key, val in pair_results.items():
                a, b = key.split("_vs_")
                disp = f"{self._short_name(a)} vs {self._short_name(b)}"
                items.append((disp, float(val.get('accuracy', 0.0))))
            items.sort(key=lambda x: x[1], reverse=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🥇 Top 5 Pairs:**")
                for name, acc in items[:5]:
                    st.metric(name, f"{acc:.3f}")
            with col2:
                st.markdown("**🥉 Bottom 5 Pairs:**")
                for name, acc in items[-5:]:
                    st.metric(name, f"{acc:.3f}")

    def _render_pair_explorer(self, pair_results: Dict):
        """Render detailed pair explorer."""
        with st.expander("🔍 Detailed Pair Analysis", expanded=False):
            if not pair_results:
                st.caption("No per-pair results available.")
                return

            # Build display names
            pair_names = list(pair_results.keys())
            pair_display_names = []
            for name in pair_names:
                a, b = name.split("_vs_")
                pair_display_names.append(f"{self._short_name(a)} vs {self._short_name(b)}")

            selected_display = st.selectbox(
                "Select pair to analyze:",
                options=pair_display_names,
                help="Choose a specific pair for detailed analysis"
            )

            if selected_display:
                # Map back to original key
                try:
                    idx = pair_display_names.index(selected_display)
                    key = pair_names[idx]
                except ValueError:
                    key = None

                if key and key in pair_results:
                    result = pair_results[key]

                    # Display pair details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{float(result.get('accuracy', 0.0)):.3f}")
                    with col2:
                        st.metric("CV Mean", f"{float(result.get('cv_mean', 0.0)):.3f}")
                    with col3:
                        st.metric("CV Std", f"{float(result.get('cv_std', 0.0)):.3f}")

                    # Confusion matrix for this pair (use shortened scenario names)
                    if 'confusion_matrix' in result and 'scenarios' in result:
                        sc = result['scenarios']
                        short_sc = self._shorten_many(sc)
                        st.markdown(f"**Confusion Matrix for {selected_display}:**")
                        cm_df = pd.DataFrame(
                            result['confusion_matrix'],
                            index=[f"True {s}" for s in short_sc],
                            columns=[f"Pred {s}" for s in short_sc]
                        )
                        st.dataframe(cm_df, use_container_width=True)

    def _render_dataset_composition(self, results: Dict, group_names: List[str]):
        """Render dataset composition visualization."""
        X_train = results.get('X_train')
        y_train = results.get('y_train')
        X_test = results.get('X_test')
        y_test = results.get('y_test')

        if X_train is None or y_train is None or X_test is None or y_test is None:
            return

        with st.expander("📊 Dataset Composition", expanded=False):
            # Calculate group distributions
            train_counts = [int(sum(1 for y in y_train if y == i)) for i in range(len(group_names))]
            test_counts = [int(sum(1 for y in y_test if y == i)) for i in range(len(group_names))]

            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Training', x=group_names, y=train_counts))
                fig.add_trace(go.Bar(name='Testing', x=group_names, y=test_counts))
                fig.update_layout(
                    title='Dataset Composition by Group',
                    xaxis_title='Groups',
                    yaxis_title='Number of Samples',
                    barmode='stack',
                    height=420
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(pd.DataFrame({"Group": group_names, "Train": train_counts, "Test": test_counts}).set_index("Group"))

    def _render_feature_importance(self, results: Dict, model_info: Dict):
        """Render feature importance if available."""
        importance = results.get('feature_importance')
        if importance is None:
            with st.expander("🎯 Feature Importance (Not Available)", expanded=False):
                st.caption("Feature importance requires linear or tree models; not all models expose it.")
            return

        feature_names = model_info.get('feature_names') or results.get('feature_names')
        if not feature_names:
            # Try to infer generic names
            feature_names = [f'Feature_{i}' for i in range(len(importance))]

        with st.expander("🎯 Feature Importance", expanded=False):
            if PLOTLY_AVAILABLE:
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
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
                importance_df = importance_df.sort_values('Importance', ascending=False)
                st.dataframe(importance_df, hide_index=True)

    def _render_classification_report_details(self, results: Dict, labels: List[str], scenario_names: List[str]):
        """Render detailed classification report."""
        st.markdown("### 📋 Detailed Classification Report")

        with st.expander("Full Classification Report", expanded=False):
            st.text(results.get('classification_report', ''))

        with st.expander("Summary", expanded=False):
            st.markdown(
                f"""
                - **Scenarios/Groups**: {', '.join(map(str, scenario_names))}
                - **Labels**: {', '.join(map(str, labels))}
                - **Model Performance**: {float(results.get('test_accuracy', 0.0)):.1%} test accuracy
                - **Cross-Validation**: {float(results.get('cv_mean', 0.0)):.3f} ± {float(results.get('cv_std', 0.0)):.3f}
                - **Training Samples**: {results.get('train_samples', results.get('total_train', 'N/A'))}
                - **Testing Samples**: {results.get('test_samples', results.get('total_test', 'N/A'))}
                """
            )


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
