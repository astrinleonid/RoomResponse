#!/usr/bin/env python3
"""
GUI Classification Panel

Handles classification interface for single pair, all pairs, and group vs group classification.
Uses existing classification logic from ScenarioClassifier.py and FeatureExtractor.py
"""

import os
import tempfile
from typing import List, Dict, Tuple, Optional, Set
import streamlit as st
import pandas as pd
import numpy as np

# Classification modules
try:
    from ScenarioClassifier import ScenarioClassifier
except ImportError:
    ScenarioClassifier = None

try:
    from FeatureExtractor import AudioFeatureExtractor
except ImportError:
    AudioFeatureExtractor = None

# Session keys for classification artifacts
SK_CLASSIFICATION_ARTIFACTS = "classification_artifacts"
SK_LAST_MODEL_INFO = "last_model_info"
SK_CLASSIFICATION_RESULTS = "classification_results"


class ClassificationPanel:
    """Classification panel implementation with three classification modes."""
    
    def __init__(self, scenario_manager):
        """Initialize with scenario manager for data access."""
        self.scenario_manager = scenario_manager
    
    def render(self):
        """Render the classification panel UI."""
        st.header("Classify - Machine Learning Classification")
        
        # Check dependencies
        if not self._check_dependencies():
            return
        
        # Get dataset root and validate
        root = st.session_state.get("dataset_root", os.getcwd())
        if not os.path.isdir(root):
            st.error("❌ Please provide a valid dataset root directory to continue.", icon="📁")
            return
        
        # Load scenarios and check for selected scenarios
        df = self._load_and_validate_scenarios(root)
        if df is None:
            return
        
        # Global classification parameters
        global_params = self._render_global_parameters()
        
        # Classification mode selection
        classification_mode = self._render_mode_selection(df)
        
        # Validate selection based on mode
        is_valid, working_df = self._validate_selection_for_mode(df, classification_mode)
        if not is_valid:
            return
        
        # Mode-specific configuration and execution
        if classification_mode == "Single Pair":
            self._render_single_pair_mode(working_df, global_params)
        elif classification_mode == "All Pairs":
            self._render_all_pairs_mode(working_df, global_params)
        else:  # Group vs Group
            self._render_group_vs_group_mode(working_df, global_params)
        
        # Model management section
        self._render_model_management()
    
    def _check_dependencies(self) -> bool:
        """Check if required classification modules are available."""
        if ScenarioClassifier is None:
            st.error("❌ ScenarioClassifier module not found. Please ensure ScenarioClassifier.py is available.", icon="🔧")
            st.info("The ScenarioClassifier.py file should be in the same directory as this GUI application.")
            return False
        return True
    
    def _load_and_validate_scenarios(self, root: str) -> Optional[pd.DataFrame]:
        """Load scenarios and validate for classification."""
        with st.spinner("Loading scenarios..."):
            df = self.scenario_manager.build_scenarios_df(root)
        
        if df.empty:
            st.info("No scenarios found in the dataset root.", icon="📂")
            return None
        
        st.success(f"✅ {len(df)} scenarios loaded from dataset")
        return df
    
    def _validate_selection_for_mode(self, df: pd.DataFrame, mode: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Validate scenario selection based on classification mode."""
        if mode == "Group vs Group":
            # Group mode doesn't require selection - uses all scenarios with labels
            return True, df
        
        # Single Pair and All Pairs require selection
        selected_paths = st.session_state.get("scenarios_selected_set", set())
        if not selected_paths:
            st.warning("⚠️ No scenarios selected. Please go to the Scenarios panel and select scenarios for classification.")
            st.info("💡 Tip: Use the Scenarios panel to select the scenarios you want to classify.")
            return False, None
        
        # Filter to selected scenarios only
        selected_df = df[df['path'].isin(selected_paths)].copy()
        if selected_df.empty:
            st.warning("⚠️ Selected scenarios not found in current dataset. Please refresh the scenarios list.")
            return False, None
        
        st.success(f"✅ {len(selected_df)} scenarios ready for classification")
        return True, selected_df
    
    def _render_global_parameters(self) -> dict:
        """Render global classification parameters."""
        st.markdown("### Global Parameters")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            feature_type = st.selectbox(
                "Feature type",
                options=["spectrum", "mfcc"],
                index=0,  # Default to MFCC
                help="Type of features to use for classification"
            )
        
        with col2:
            model_type = st.selectbox(
                "Model type",
                options=["svm", "logistic"],
                index=0,  # Default to SVM
                help="Machine learning algorithm to use"
            )
        
        with col3:
            test_size = st.slider(
                "Test size",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Fraction of data to use for testing (rest for training)"
            )
        
        with col4:
            cv_folds = st.slider(
                "CV folds",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                help="Number of cross-validation folds"
            )
        
        return {
            "feature_type": feature_type,
            "model_type": model_type,
            "test_size": test_size,
            "cv_folds": cv_folds
        }
    
    def _render_mode_selection(self, df: pd.DataFrame) -> str:
        """Render classification mode selection."""
        st.markdown("### Classification Mode")
        
        # Count available options
        selected_paths = st.session_state.get("scenarios_selected_set", set())
        selected_count = len([path for path in selected_paths if path in df['path'].values])
        unique_labels = self.scenario_manager.get_unique_labels(df)
        
        # Mode options with dynamic enabling
        mode_options = ["Single Pair", "All Pairs", "Group vs Group"]
        
        # Show mode info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Single Pair**")
            if selected_count == 2:
                st.success(f"✅ Ready ({selected_count} selected)")
            else:
                st.error(f"❌ Needs exactly 2 selected ({selected_count} selected)")
        
        with col2:
            st.markdown("**All Pairs**")
            if selected_count >= 2:
                st.success(f"✅ Ready ({selected_count} selected)")
            else:
                st.error(f"❌ Needs 2+ selected ({selected_count} selected)")
        
        with col3:
            st.markdown("**Group vs Group**")
            if len(unique_labels) >= 2:
                st.success(f"✅ Ready ({len(unique_labels)} groups)")
            else:
                st.error(f"❌ Needs 2+ groups ({len(unique_labels)} groups)")
        
        # Determine default mode based on data
        default_index = 2  # Default to Group vs Group
        if selected_count == 2:
            default_index = 0  # Single Pair
        elif selected_count > 2:
            default_index = 1  # All Pairs
        elif len(unique_labels) >= 2:
            default_index = 2  # Group vs Group
        
        mode = st.radio(
            "Select classification mode:",
            options=mode_options,
            index=default_index,
            help="Choose how to perform classification based on your data"
        )
        
        # Show mode-specific requirements
        if mode == "Single Pair":
            st.info("💡 **Single Pair**: Requires exactly 2 scenarios selected in the Scenarios panel")
        elif mode == "All Pairs":
            st.info("💡 **All Pairs**: Requires 2+ scenarios selected in the Scenarios panel")
        else:  # Group vs Group
            st.info("💡 **Group vs Group**: Uses ALL scenarios with group labels (no selection required)")
        
        return mode
    
    def _render_single_pair_mode(self, df: pd.DataFrame, global_params: dict):
        """Render single pair classification interface."""
        st.markdown("### Single Pair Classification")
        
        if len(df) != 2:
            st.error(f"❌ Single Pair mode requires exactly 2 scenarios. You have {len(df)} selected.")
            st.info("💡 Go to the Scenarios panel and select exactly 2 scenarios.")
            return
        
        # Show the two scenarios
        scenario_list = df['scenario'].tolist()
        scenario_paths = df['path'].tolist()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Scenario A:** {scenario_list[0]}")
            features_a = self.scenario_manager.check_features_available(scenario_paths[0])
            self._show_feature_status(features_a, global_params["feature_type"])
        
        with col2:
            st.markdown(f"**Scenario B:** {scenario_list[1]}")
            features_b = self.scenario_manager.check_features_available(scenario_paths[1])
            self._show_feature_status(features_b, global_params["feature_type"])
        
        # Check if both scenarios have required features
        feature_available_a = features_a.get(global_params["feature_type"], False)
        feature_available_b = features_b.get(global_params["feature_type"], False)
        
        if not (feature_available_a and feature_available_b):
            st.error(f"❌ Both scenarios must have {global_params['feature_type']} features. Use the Process panel to extract features first.")
            return
        
        # Custom labels for binary classification
        st.markdown("### Classification Labels")
        label_col1, label_col2 = st.columns([1, 1])
        
        with label_col1:
            # Extract scenario number for default label
            number_a = df.iloc[0].get('number_str', '1')
            label_a = st.text_input(
                f"Label for {scenario_list[0]}",
                value=f"Scenario_{number_a}",
                help="Custom label for classification"
            )
        
        with label_col2:
            number_b = df.iloc[1].get('number_str', '2')
            label_b = st.text_input(
                f"Label for {scenario_list[1]}",
                value=f"Scenario_{number_b}",
                help="Custom label for classification"
            )
        
        # Execute classification
        st.markdown("### Execute Classification")
        
        if st.button("🚀 Run Single Pair Classification", type="primary", use_container_width=True):
            if label_a.strip() and label_b.strip() and label_a.strip() != label_b.strip():
                SinglePairExecutor(self.scenario_manager).execute(
                    scenario_paths=scenario_paths,
                    scenario_names=scenario_list,
                    labels=[label_a.strip(), label_b.strip()],
                    global_params=global_params
                )
            else:
                st.error("❌ Please provide distinct, non-empty labels for both scenarios.")
    
    def _render_all_pairs_mode(self, df: pd.DataFrame, global_params: dict):
        """Render all pairs classification interface."""
        st.markdown("### All Pairs Classification")
        
        if len(df) < 2:
            st.error(f"❌ All Pairs mode requires at least 2 scenarios. You have {len(df)} selected.")
            return
        
        # Check feature availability for all scenarios
        missing_features = []
        scenario_info = []
        
        for _, row in df.iterrows():
            features = self.scenario_manager.check_features_available(row['path'])
            has_features = features.get(global_params["feature_type"], False)
            scenario_info.append({
                'name': row['scenario'],
                'path': row['path'],
                'has_features': has_features
            })
            if not has_features:
                missing_features.append(row['scenario'])
        
        # Show scenario overview
        st.markdown(f"**Selected scenarios:** {len(df)}")
        st.markdown(f"**Total pairs to classify:** {len(df) * (len(df) - 1) // 2}")
        
        if missing_features:
            st.error(f"❌ Missing {global_params['feature_type']} features for: {', '.join(missing_features)}")
            st.info("Use the Process panel to extract features for all scenarios first.")
            return
        
        # Preview pairs
        with st.expander("Preview classification pairs"):
            pairs = [(scenario_info[i]['name'], scenario_info[j]['name']) 
                    for i in range(len(scenario_info)) 
                    for j in range(i+1, len(scenario_info))]
            
            for i, (name_a, name_b) in enumerate(pairs[:10]):  # Show first 10 pairs
                st.text(f"{i+1:2d}. {name_a} vs {name_b}")
            
            if len(pairs) > 10:
                st.caption(f"... and {len(pairs) - 10} more pairs")
        
        # Execute classification
        st.markdown("### Execute Classification")
        
        if st.button("🚀 Run All Pairs Classification", type="primary", use_container_width=True):
            AllPairsExecutor(self.scenario_manager).execute(
                scenario_info=scenario_info,
                global_params=global_params
            )
    
    def _render_group_vs_group_mode(self, df: pd.DataFrame, global_params: dict):
        """Render group vs group classification interface."""
        st.markdown("### Group vs Group Classification")
        
        # Get unique labels
        unique_labels = self.scenario_manager.get_unique_labels(df)
        
        if len(unique_labels) < 2:
            st.error(f"❌ Group vs Group mode requires at least 2 different group labels. Found: {len(unique_labels)}")
            if len(unique_labels) == 0:
                st.info("💡 Use the Scenarios panel to assign group labels to your scenarios.")
            return
        
        # Show group distribution
        st.markdown("### Group Distribution")
        
        group_counts = {}
        scenarios_by_group = {}
        
        for _, row in df.iterrows():
            label = row.get('label', '') or ''
            # Split labels and clean them
            labels = [l.strip() for l in str(label).split(',') if l.strip()]
            
            if not labels:
                labels = ['unlabeled']
            
            for lbl in labels:
                if lbl not in group_counts:
                    group_counts[lbl] = 0
                    scenarios_by_group[lbl] = []
                group_counts[lbl] += 1
                scenarios_by_group[lbl].append(row['scenario'])
        
        # Display group information
        for group_label, count in sorted(group_counts.items()):
            st.markdown(f"**{group_label}:** {count} scenarios")
            if count < 2:
                st.warning(f"⚠️ Group '{group_label}' has only {count} scenario(s). Need at least 2 for classification.")
        
        # Group selection for classification
        st.markdown("### Select Groups to Classify")
        
        valid_groups = [label for label, count in group_counts.items() if count >= 2]
        
        if len(valid_groups) < 2:
            st.error("❌ Need at least 2 groups with 2+ scenarios each for group classification.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            group_a = st.selectbox(
                "Group A",
                options=valid_groups,
                index=0,
                help="First group for classification"
            )
        
        with col2:
            available_groups_b = [g for g in valid_groups if g != group_a]
            group_b = st.selectbox(
                "Group B", 
                options=available_groups_b,
                index=0 if available_groups_b else None,
                help="Second group for classification"
            )
        
        if not group_b:
            st.error("❌ Please select two different groups.")
            return
        
        # Show selected groups info
        scenarios_a = scenarios_by_group[group_a]
        scenarios_b = scenarios_by_group[group_b]
        
        st.info(f"**{group_a}:** {len(scenarios_a)} scenarios | **{group_b}:** {len(scenarios_b)} scenarios")
        
        # Get scenario paths for selected groups
        scenarios_a_paths = []
        scenarios_b_paths = []
        
        for _, row in df.iterrows():
            row_labels = str(row.get('label', '')).split(',')
            row_labels = [l.strip() for l in row_labels if l.strip()]
            
            if group_a in row_labels:
                scenarios_a_paths.append(row['path'])
            if group_b in row_labels:
                scenarios_b_paths.append(row['path'])
        
        # Check feature availability
        missing_features = []
        for path in scenarios_a_paths + scenarios_b_paths:
            features = self.scenario_manager.check_features_available(path)
            if not features.get(global_params["feature_type"], False):
                scenario_name = os.path.basename(path)
                missing_features.append(scenario_name)
        
        if missing_features:
            st.error(f"❌ Missing {global_params['feature_type']} features for: {', '.join(missing_features)}")
            st.info("Use the Process panel to extract features for all scenarios first.")
            return
        
        # Execute classification
        st.markdown("### Execute Classification")
        
        if st.button("🚀 Run Group vs Group Classification", type="primary", use_container_width=True):
            GroupVsGroupExecutor(self.scenario_manager).execute(
                group_a=group_a,
                group_b=group_b,
                scenarios_a_paths=scenarios_a_paths,
                scenarios_b_paths=scenarios_b_paths,
                global_params=global_params
            )
    
    def _show_feature_status(self, features: dict, required_type: str):
        """Show feature availability status."""
        if features.get(required_type, False):
            st.success(f"✅ {required_type.upper()} features available")
        else:
            st.error(f"❌ {required_type.upper()} features missing")
    
    def _render_model_management(self):
        """Render model management section."""
        st.markdown("---")
        st.markdown("### Model Management")
        
        # Check if we have a trained model in session
        model_info = st.session_state.get(SK_LAST_MODEL_INFO)
        classification_artifacts = st.session_state.get(SK_CLASSIFICATION_ARTIFACTS)
        
        if model_info and classification_artifacts:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.success(f"✅ Model trained: {model_info['model_type'].upper()} with {model_info['feature_type']} features")
                st.caption(f"Accuracy: {model_info.get('test_accuracy', 0):.3f} | Mode: {model_info.get('mode', 'Unknown')}")
            
            with col2:
                if st.button("💾 Download Model", help="Download trained model file"):
                    self._download_model()
            
            with col3:
                if st.button("🗑️ Clear Model", help="Clear current model from session"):
                    st.session_state[SK_CLASSIFICATION_ARTIFACTS] = None
                    st.session_state[SK_LAST_MODEL_INFO] = None
                    st.session_state[SK_CLASSIFICATION_RESULTS] = None
                    st.success("Model cleared from session")
                    st.rerun()
        else:
            st.info("💡 No trained model in current session. Train a model using one of the classification modes above.")
        
        # Model upload for prediction
        st.markdown("### Upload Trained Model")
        uploaded_model = st.file_uploader(
            "Upload model file",
            type=['joblib', 'pkl'],
            help="Upload a previously trained model for use in prediction"
        )
        
        if uploaded_model is not None:
            try:
                model_bytes = uploaded_model.read()
                classifier = ScenarioClassifier.loads_model_bytes(model_bytes)
                
                # Store in session for prediction panel
                st.session_state[SK_CLASSIFICATION_ARTIFACTS] = model_bytes
                st.session_state[SK_LAST_MODEL_INFO] = {
                    'model_type': classifier.model_type,
                    'feature_type': classifier.feature_type,
                    'mode': 'uploaded',
                    'feature_names': classifier.feature_names,
                    'label_names': list(classifier.label_encoder.classes_) if classifier.label_encoder else []
                }
                
                st.success(f"✅ Model uploaded: {classifier.model_type.upper()} with {classifier.feature_type} features")
                
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
    
    def _download_model(self):
        """Provide model download functionality."""
        model_bytes = st.session_state.get(SK_CLASSIFICATION_ARTIFACTS)
        model_info = st.session_state.get(SK_LAST_MODEL_INFO)
        
        if model_bytes and model_info:
            filename = f"room_response_model_{model_info['model_type']}_{model_info['feature_type']}.joblib"
            
            st.download_button(
                label="📥 Download Model File",
                data=model_bytes,
                file_name=filename,
                mime="application/octet-stream",
                help="Download the trained model for later use"
            )


class SinglePairExecutor:
    """Handles single pair classification execution."""
    
    def __init__(self, scenario_manager):
        self.scenario_manager = scenario_manager
    
    def execute(self, scenario_paths: List[str], scenario_names: List[str], 
               labels: List[str], global_params: dict):
        """Execute single pair classification using ScenarioClassifier API."""
        st.markdown("### 🔬 Running Single Pair Classification")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize classifier
            status_text.text("Initializing classifier...")
            progress_bar.progress(10)
            
            classifier = ScenarioClassifier(
                model_type=global_params["model_type"],
                feature_type=global_params["feature_type"]
            )
            
            # Prepare dataset using ScenarioClassifier's built-in method
            status_text.text("Loading and preparing dataset...")
            progress_bar.progress(30)
            
            X, y, feature_names, label_names = classifier.prepare_dataset(
                scenario_paths[0], scenario_paths[1],
                labels[0], labels[1]
            )
            
            st.info(f"📊 Dataset prepared: {len(X)} samples, {len(feature_names)} features")
            
            # Train and evaluate
            status_text.text("Training and evaluating model...")
            progress_bar.progress(60)
            
            results = classifier.train_and_evaluate(
                X, y,
                test_size=global_params["test_size"],
                cv_folds=global_params["cv_folds"]
            )
            
            progress_bar.progress(90)
            
            # Store results in session
            model_bytes = classifier.dumps_model_bytes(extra_meta={
                'scenario_names': scenario_names,
                'labels': labels,
                'global_params': global_params
            })
            
            st.session_state[SK_CLASSIFICATION_ARTIFACTS] = model_bytes
            st.session_state[SK_LAST_MODEL_INFO] = {
                'model_type': global_params["model_type"],
                'feature_type': global_params["feature_type"],
                'mode': 'single_pair',
                'scenario_names': scenario_names,
                'labels': labels,
                'test_accuracy': results['test_accuracy'],
                'feature_names': feature_names,
                'label_names': label_names
            }
            st.session_state[SK_CLASSIFICATION_RESULTS] = results
            
            progress_bar.progress(100)
            status_text.text("Classification complete!")
            
            # Display results
            self._display_single_pair_results(results, scenario_names, labels)
            
            # Enable visualization
            st.success("🎉 Classification completed! You can now use the Predict and Visualize panels.")
            
        except Exception as e:
            st.error(f"❌ Classification failed: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
            progress_bar.empty()
            status_text.empty()
    
    def _display_single_pair_results(self, results: dict, scenario_names: List[str], labels: List[str]):
        """Display classification results."""
        st.markdown("### 📊 Classification Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train Accuracy", f"{results['train_accuracy']:.3f}")
        
        with col2:
            st.metric("Test Accuracy", f"{results['test_accuracy']:.3f}")
        
        with col3:
            st.metric("CV Mean", f"{results['cv_mean']:.3f}")
        
        with col4:
            st.metric("CV Std", f"{results['cv_std']:.3f}")
        
        # Confusion matrix
        st.markdown("### Confusion Matrix")
        cm = results['confusion_matrix']
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        st.dataframe(cm_df, use_container_width=True)
        
        # Classification report
        with st.expander("📋 Detailed Classification Report"):
            st.text(results['classification_report'])
        
        # Cross-validation scores
        with st.expander("📈 Cross-Validation Scores"):
            cv_scores = results['cv_scores']
            for i, score in enumerate(cv_scores):
                st.text(f"Fold {i+1}: {score:.3f}")
            st.text(f"Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")


class AllPairsExecutor:
    """Handles all pairs classification execution."""
    
    def __init__(self, scenario_manager):
        self.scenario_manager = scenario_manager
    
    def execute(self, scenario_info: List[dict], global_params: dict):
        """Execute all pairs classification using ScenarioClassifier API."""
        st.markdown("### 🔬 Running All Pairs Classification")
        
        total_pairs = len(scenario_info) * (len(scenario_info) - 1) // 2
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize results storage
            results_matrix = np.zeros((len(scenario_info), len(scenario_info)))
            pair_results = {}
            
            status_text.text(f"Starting classification of {total_pairs} pairs...")
            
            pair_count = 0
            
            # Process all pairs
            for i in range(len(scenario_info)):
                for j in range(i + 1, len(scenario_info)):
                    scenario_a = scenario_info[i]
                    scenario_b = scenario_info[j]
                    
                    pair_count += 1
                    status_text.text(f"Classifying pair {pair_count}/{total_pairs}: {scenario_a['name']} vs {scenario_b['name']}")
                    
                    # Create temporary labels for this pair
                    label_a = f"Scenario_{i}"
                    label_b = f"Scenario_{j}"
                    
                    # Initialize classifier for this pair
                    classifier = ScenarioClassifier(
                        model_type=global_params["model_type"],
                        feature_type=global_params["feature_type"]
                    )
                    
                    # Prepare dataset for this pair using ScenarioClassifier's method
                    X, y, feature_names, label_names = classifier.prepare_dataset(
                        scenario_a['path'], scenario_b['path'],
                        label_a, label_b
                    )
                    
                    # Train and evaluate
                    results = classifier.train_and_evaluate(
                        X, y,
                        test_size=global_params["test_size"],
                        cv_folds=global_params["cv_folds"]
                    )
                    
                    # Store results
                    accuracy = results['test_accuracy']
                    results_matrix[i, j] = accuracy
                    results_matrix[j, i] = accuracy  # symmetric
                    
                    pair_key = f"{scenario_a['name']}_vs_{scenario_b['name']}"
                    pair_results[pair_key] = {
                        'accuracy': accuracy,
                        'cv_mean': results['cv_mean'],
                        'cv_std': results['cv_std'],
                        'confusion_matrix': results['confusion_matrix'],
                        'scenarios': [scenario_a['name'], scenario_b['name']]
                    }
                    
                    progress_bar.progress(pair_count / total_pairs)
            
            # Store comprehensive results
            all_pairs_results = {
                'accuracy_matrix': results_matrix,
                'pair_results': pair_results,
                'scenario_names': [info['name'] for info in scenario_info],
                'global_params': global_params,
                'total_pairs': total_pairs
            }
            
            st.session_state[SK_CLASSIFICATION_RESULTS] = all_pairs_results
            st.session_state[SK_LAST_MODEL_INFO] = {
                'model_type': global_params["model_type"],
                'feature_type': global_params["feature_type"],
                'mode': 'all_pairs',
                'total_pairs': total_pairs,
                'scenario_count': len(scenario_info)
            }
            
            status_text.text("All pairs classification complete!")
            
            # Display results
            self._display_all_pairs_results(all_pairs_results)
            
            st.success("🎉 All pairs classification completed! You can now use the Visualize panel.")
            
        except Exception as e:
            st.error(f"❌ All pairs classification failed: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
            progress_bar.empty()
            status_text.empty()
    
    def _display_all_pairs_results(self, results: dict):
        """Display all pairs classification results."""
        st.markdown("### 📊 All Pairs Classification Results")
        
        accuracy_matrix = results['accuracy_matrix']
        scenario_names = results['scenario_names']
        
        # Summary statistics
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
        
        # Accuracy matrix display
        st.markdown("### Accuracy Matrix")
        
        # Create a symmetric display matrix
        display_matrix = accuracy_matrix.copy()
        np.fill_diagonal(display_matrix, 1.0)  # Perfect accuracy on diagonal
        
        matrix_df = pd.DataFrame(display_matrix, index=scenario_names, columns=scenario_names)
        st.dataframe(matrix_df.style.format("{:.3f}"), use_container_width=True)
        
        # Top and bottom performing pairs
        with st.expander("🏆 Best and Worst Performing Pairs"):
            pair_accuracies = [(k, v['accuracy']) for k, v in results['pair_results'].items()]
            pair_accuracies.sort(key=lambda x: x[1], reverse=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 5 Pairs:**")
                for pair_name, acc in pair_accuracies[:5]:
                    st.text(f"{pair_name.replace('_vs_', ' vs ')}: {acc:.3f}")
            
            with col2:
                st.markdown("**Bottom 5 Pairs:**")
                for pair_name, acc in pair_accuracies[-5:]:
                    st.text(f"{pair_name.replace('_vs_', ' vs ')}: {acc:.3f}")


class GroupVsGroupExecutor:
    """Handles group vs group classification execution."""
    
    def __init__(self, scenario_manager):
        self.scenario_manager = scenario_manager
    
    def execute(self, group_a: str, group_b: str, scenarios_a_paths: List[str], 
               scenarios_b_paths: List[str], global_params: dict):
        """Execute group vs group classification using ScenarioClassifier API."""
        st.markdown("### 🔬 Running Group vs Group Classification")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # We'll use ScenarioClassifier by combining all scenarios from each group
            # into temporary merged folders, then using the standard API
            
            status_text.text("Preparing group data...")
            progress_bar.progress(20)
            
            # Create temporary directories for merged group data
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_group_a = os.path.join(temp_dir, f"group_{group_a}")
                temp_group_b = os.path.join(temp_dir, f"group_{group_b}")
                
                os.makedirs(temp_group_a, exist_ok=True)
                os.makedirs(temp_group_b, exist_ok=True)
                
                # Merge features from all scenarios in each group
                self._merge_group_features(scenarios_a_paths, temp_group_a, global_params["feature_type"])
                self._merge_group_features(scenarios_b_paths, temp_group_b, global_params["feature_type"])
                
                progress_bar.progress(40)
                
                # Initialize classifier
                status_text.text("Initializing classifier...")
                
                classifier = ScenarioClassifier(
                    model_type=global_params["model_type"],
                    feature_type=global_params["feature_type"]
                )
                
                # Prepare dataset using merged group data
                status_text.text("Preparing dataset...")
                progress_bar.progress(60)
                
                X, y, feature_names, label_names = classifier.prepare_dataset(
                    temp_group_a, temp_group_b,
                    group_a, group_b
                )
                
                st.info(f"📊 Combined dataset: {len(X)} samples from {len(scenarios_a_paths) + len(scenarios_b_paths)} scenarios")
                
                # Train and evaluate
                status_text.text("Training and evaluating model...")
                progress_bar.progress(80)
                
                results = classifier.train_and_evaluate(
                    X, y,
                    test_size=global_params["test_size"],
                    cv_folds=global_params["cv_folds"]
                )
                
                progress_bar.progress(95)
                
                # Store results
                model_bytes = classifier.dumps_model_bytes(extra_meta={
                    'group_a': group_a,
                    'group_b': group_b,
                    'scenarios_a_count': len(scenarios_a_paths),
                    'scenarios_b_count': len(scenarios_b_paths),
                    'global_params': global_params
                })
                
                st.session_state[SK_CLASSIFICATION_ARTIFACTS] = model_bytes
                st.session_state[SK_LAST_MODEL_INFO] = {
                    'model_type': global_params["model_type"],
                    'feature_type': global_params["feature_type"],
                    'mode': 'group_vs_group',
                    'group_a': group_a,
                    'group_b': group_b,
                    'test_accuracy': results['test_accuracy'],
                    'feature_names': feature_names,
                    'label_names': label_names
                }
                st.session_state[SK_CLASSIFICATION_RESULTS] = results
                
                progress_bar.progress(100)
                status_text.text("Classification complete!")
                
                # Display results
                self._display_group_results(results, group_a, group_b)
                
                st.success("🎉 Group classification completed! You can now use the Predict and Visualize panels.")
            
        except Exception as e:
            st.error(f"❌ Group classification failed: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
            progress_bar.empty()
            status_text.empty()
    
    def _merge_group_features(self, scenario_paths: List[str], output_dir: str, feature_type: str):
        """Merge feature files from multiple scenarios into a single directory."""
        # Determine the correct feature filename
        if feature_type == "spectrum":
            feature_filename = "spectrum.csv"
        else:
            feature_filename = "features.csv"
        
        all_rows = []
        
        # Read and combine all feature files
        for scenario_path in scenario_paths:
            feature_file = os.path.join(scenario_path, feature_filename)
            if os.path.exists(feature_file):
                try:
                    df = pd.read_csv(feature_file)
                    # Add scenario identifier to filename for uniqueness
                    scenario_name = os.path.basename(scenario_path)
                    if 'filename' in df.columns:
                        df['filename'] = df['filename'].apply(lambda x: f"{scenario_name}_{x}")
                    all_rows.append(df)
                except Exception as e:
                    st.warning(f"Failed to read {feature_file}: {e}")
        
        if all_rows:
            # Combine all dataframes
            merged_df = pd.concat(all_rows, ignore_index=True)
            
            # Save merged features
            output_file = os.path.join(output_dir, feature_filename)
            merged_df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"No valid feature files found for {feature_type}")
    
    def _display_group_results(self, results: dict, group_a: str, group_b: str):
        """Display group classification results with detailed dataset statistics."""
        st.markdown("### 📊 Group Classification Results")
        
        # Extract dataset statistics from results
        X_train = results.get('X_train')
        X_test = results.get('X_test') 
        y_train = results.get('y_train')
        y_test = results.get('y_test')
        
        # Show detailed dataset statistics
        st.markdown("### 📈 Dataset Statistics")
        
        if X_train is not None and y_train is not None:
            # Calculate train/test split details
            total_samples = len(X_train) + len(X_test)
            train_group_a = sum(1 for y in y_train if y == 0)  # Assuming 0 is first group
            train_group_b = sum(1 for y in y_train if y == 1)  # Assuming 1 is second group
            test_group_a = sum(1 for y in y_test if y == 0)
            test_group_b = sum(1 for y in y_test if y == 1)
            
            # Display statistics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", total_samples)
                st.caption(f"Train: {len(X_train)} | Test: {len(X_test)}")
            
            with col2:
                st.metric(f"{group_a} Samples", train_group_a + test_group_a)
                st.caption(f"Train: {train_group_a} | Test: {test_group_a}")
            
            with col3:
                st.metric(f"{group_b} Samples", train_group_b + test_group_b)
                st.caption(f"Train: {train_group_b} | Test: {test_group_b}")
            
            with col4:
                test_ratio = len(X_test) / total_samples if total_samples > 0 else 0
                st.metric("Test Ratio", f"{test_ratio:.1%}")
                st.caption(f"As configured: {results.get('test_size', 'unknown')}")
            
            # Detailed breakdown table
            with st.expander("📋 Detailed Dataset Breakdown"):
                breakdown_data = {
                    "Group": [group_a, group_b, "Total"],
                    "Train Samples": [train_group_a, train_group_b, len(X_train)],
                    "Test Samples": [test_group_a, test_group_b, len(X_test)],
                    "Total Samples": [train_group_a + test_group_a, train_group_b + test_group_b, total_samples],
                    "Train %": [f"{train_group_a/len(X_train)*100:.1f}%" if len(X_train) > 0 else "0%",
                               f"{train_group_b/len(X_train)*100:.1f}%" if len(X_train) > 0 else "0%",
                               "100%"],
                    "Test %": [f"{test_group_a/len(X_test)*100:.1f}%" if len(X_test) > 0 else "0%",
                              f"{test_group_b/len(X_test)*100:.1f}%" if len(X_test) > 0 else "0%",
                              "100%"]
                }
                
                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                
                # Additional info
                st.caption(f"**Features per sample:** {X_train.shape[1] if X_train.ndim > 1 else 'N/A'}")
                st.caption(f"**Stratified split:** Yes (maintains group proportions)")
        
        # Key metrics
        st.markdown("### 🎯 Classification Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train Accuracy", f"{results['train_accuracy']:.3f}")
        
        with col2:
            st.metric("Test Accuracy", f"{results['test_accuracy']:.3f}")
        
        with col3:
            st.metric("CV Mean", f"{results['cv_mean']:.3f}")
        
        with col4:
            st.metric("CV Std", f"{results['cv_std']:.3f}")
        
        # Confusion matrix
        st.markdown("### 🔀 Confusion Matrix")
        cm = results['confusion_matrix']
        cm_df = pd.DataFrame(cm, index=[group_a, group_b], columns=[group_a, group_b])
        
        # Add totals and percentages
        cm_with_totals = cm_df.copy()
        cm_with_totals['Total'] = cm_df.sum(axis=1)
        totals_row = cm_df.sum(axis=0).tolist() + [cm_df.sum().sum()]
        cm_with_totals.loc['Total'] = totals_row
        
        st.dataframe(cm_with_totals, use_container_width=True)
        
        # Confusion matrix explanation
        st.caption(f"""
        **Reading the matrix:** 
        - Rows = True labels, Columns = Predicted labels
        - Diagonal values = Correct predictions
        - Off-diagonal = Misclassifications
        """)
        
        # Classification report
        with st.expander("📋 Detailed Classification Report"):
            st.text(results['classification_report'])
        
        # Cross-validation scores
        with st.expander("📈 Cross-Validation Scores"):
            cv_scores = results['cv_scores']
            for i, score in enumerate(cv_scores):
                st.text(f"Fold {i+1}: {score:.3f}")
            st.text(f"Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            st.caption("**Cross-validation:** Uses the full dataset with stratified folds to estimate model performance.")