#!/usr/bin/env python3
"""
Classification Panel (Refactored & Caching)

- Delegates ML logic to ScenarioClassifier
- Stores results in session state for Visualization panel:
    SK_CLASSIFICATION_RESULTS
    SK_LAST_MODEL_INFO
    SK_CLASSIFICATION_ARTIFACTS
- Compact 'Details' expanders for all modes
"""
import os
from typing import List, Tuple, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np

try:
    from ScenarioClassifier import ScenarioClassifier
except ImportError:
    ScenarioClassifier = None

# Shared keys (must match gui_visualize_panel expectations)
SK_DATASET_ROOT = "dataset_root"
SK_SCN_SELECTIONS = "scenarios_selected_set"
SK_CLASSIFIER_OBJ = "classifier_obj"

SK_CLASSIFICATION_RESULTS = "classification_results"
SK_LAST_MODEL_INFO = "last_model_info"
SK_CLASSIFICATION_ARTIFACTS = "classification_artifacts"


class ClassificationPanel:
    def __init__(self, scenario_manager):
        self.scenario_manager = scenario_manager

    # ----------------- main render -----------------
    def render(self):
        st.header("Classify — Machine Learning")
        if ScenarioClassifier is None:
            st.error("❌ ScenarioClassifier.py not found.")
            return

        root = st.session_state.get(SK_DATASET_ROOT, os.getcwd())
        if not os.path.isdir(root):
            st.error("❌ Provide a valid dataset root directory.")
            return

        with st.spinner("Loading scenarios…"):
            df = self.scenario_manager.build_scenarios_df(root)
        if df is None or df.empty:
            st.info("No scenarios found.")
            return

        # Ensure classifier in session
        clf = st.session_state.get(SK_CLASSIFIER_OBJ)
        if clf is None or not isinstance(clf, ScenarioClassifier):
            clf = ScenarioClassifier()  # defaults inside class
            st.session_state[SK_CLASSIFIER_OBJ] = clf

        # Global params
        params = self._render_global_parameters(clf)

        # Mode selection and execution
        mode = self._render_mode_status(df)
        if mode == "Single Pair":
            self._render_single_pair(df, clf, params, root)
        elif mode == "All Pairs":
            self._render_all_pairs(df, clf, params, root)
        else:
            self._render_group_vs_group(df, clf, params, root)

        # Model management UI
        self._render_model_management(clf, root)

    # ----------------- UI helpers -----------------
    def _render_global_parameters(self, clf: ScenarioClassifier) -> dict:
        st.markdown("### Global Parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            feature_type = st.selectbox("Feature type", options=["spectrum", "mfcc"], index=1)
        with c2:
            model_type = st.selectbox("Model type", options=["svm", "logistic"], index=0)
        with c3:
            test_size = st.slider("Test size", 0.05, 0.5, 0.2, 0.05)
        with c4:
            cv_folds = st.slider("CV folds", 2, 10, 5, 1)

        # update classifier defaults
        clf.model_type = model_type
        clf.feature_type = feature_type
        return {
            "feature_type": feature_type,
            "model_type": model_type,
            "test_size": test_size,
            "cv_folds": cv_folds,
        }

    def _render_mode_status(self, df: pd.DataFrame) -> str:
        st.markdown("### Classification Mode")
        selected_paths = st.session_state.get(SK_SCN_SELECTIONS, set())
        selected_df = df[df["path"].isin(selected_paths)]
        selected_count = len(selected_df)
        unique_labels = self.scenario_manager.get_unique_labels(df)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Single Pair**")
            st.success(
                f"✅ Ready ({selected_count} selected)"
                if selected_count == 2
                else f"❌ Needs exactly 2 (have {selected_count})"
            )
            sp_ready = selected_count == 2
        with c2:
            st.markdown("**All Pairs**")
            st.success(
                f"✅ Ready ({selected_count} selected)"
                if selected_count >= 2
                else f"❌ Needs 2+ (have {selected_count})"
            )
            ap_ready = selected_count >= 2
        with c3:
            st.markdown("**Group vs Group**")
            st.success(
                f"✅ Ready ({len(unique_labels)} groups)"
                if len(unique_labels) >= 2
                else f"❌ Needs 2+ groups (have {len(unique_labels)})"
            )
            gv_ready = len(unique_labels) >= 2

        default_idx = 0 if sp_ready else (1 if ap_ready else 2)
        return st.radio(
            "Select classification mode:",
            ["Single Pair", "All Pairs", "Group vs Group"],
            index=default_idx,
        )

    # ----------------- Modes -----------------
    def _render_single_pair(self, df: pd.DataFrame, clf: ScenarioClassifier, params: dict, dataset_root: str):
        st.markdown("### Single Pair")
        selected_paths = st.session_state.get(SK_SCN_SELECTIONS, set())
        selected_df = df[df["path"].isin(selected_paths)]
        if len(selected_df) != 2:
            st.error("Select exactly 2 scenarios in the Scenarios panel.")
            return

        pairs: List[Tuple[str, str]] = selected_df[["path", "scenario"]].values.tolist()
        scen_names = [pairs[0][1], pairs[1][1]]
        st.info(f"**Scenarios:** {scen_names[0]} vs {scen_names[1]}")

        c1, c2 = st.columns(2)
        with c1:
            label_a = st.text_input("Label for first scenario", value=scen_names[0])
        with c2:
            label_b = st.text_input("Label for second scenario", value=scen_names[1])

        if st.button("🚀 Run Single Pair Classification", type="primary"):
            try:
                out = clf.run_single_pair(
                    path_a=pairs[0][0],
                    path_b=pairs[1][0],
                    label_a=label_a.strip() or "A",
                    label_b=label_b.strip() or "B",
                    params=params,
                    dataset_root=dataset_root,
                )
                metrics: Dict[str, Any] = out["metrics"]

                # ---- Cache for Visualization panel ----
                st.session_state[SK_CLASSIFICATION_RESULTS] = {
                    **metrics,
                    # helpful aliases for viz details
                    "label_names": metrics.get("label_names", [label_a, label_b]),
                    "scenario_names": scen_names,
                }
                st.session_state[SK_LAST_MODEL_INFO] = clf.get_model_info()
                st.session_state[SK_CLASSIFICATION_ARTIFACTS] = {
                    "feature_names": out.get("feature_names", []),
                    "mode": "single_pair",
                }

                st.success(f"Done. Test accuracy: {metrics.get('test_accuracy', 0):.3f}")
                with st.expander("Details"):
                    st.caption("Brief summary. Full diagnostics in the Visualization panel.")
                    st.text(metrics.get("classification_report", ""))

            except Exception as e:
                st.error(f"Failed: {e}")

    def _render_all_pairs(self, df: pd.DataFrame, clf: ScenarioClassifier, params: dict, dataset_root: str):
        st.markdown("### All Pairs")
        selected_paths = st.session_state.get(SK_SCN_SELECTIONS, set())
        selected_df = df[df["path"].isin(selected_paths)]
        if len(selected_df) < 2:
            st.error("Select 2+ scenarios in the Scenarios panel.")
            return

        scenarios = selected_df[["path", "scenario"]].values.tolist()
        st.info(f"Selected scenarios: {len(scenarios)}")
        max_samples = st.number_input(
            "Max samples per scenario", min_value=10, max_value=1000, value=100
        )

        if st.button("🚀 Run All Pairs", type="primary"):
            try:
                out = clf.run_all_pairs(
                    scenarios=scenarios,
                    params={**params, "max_samples_per_scenario": int(max_samples)},
                    dataset_root=dataset_root,
                )

                # ---- Cache for Visualization panel ----
                st.session_state[SK_CLASSIFICATION_RESULTS] = {
                    "accuracy_matrix": out.get("accuracy_matrix"),
                    "pair_results": out.get("pair_results", {}),
                    "scenario_names": out.get("scenario_names", []),
                    "summary": out.get("summary", {}),
                }
                st.session_state[SK_LAST_MODEL_INFO] = clf.get_model_info()
                st.session_state[SK_CLASSIFICATION_ARTIFACTS] = {
                    "mode": "all_pairs",
                }

                st.success(f"Completed. Pairs: {out['summary']['total_pairs']}")

                # ---- Compact Details expander ----
                with st.expander("Details"):
                    st.caption(
                        "Quick snapshot. Explore full heatmap and per-pair breakdowns in Visualization."
                    )
                    names = out.get("scenario_names", [])
                    pair_results = out.get("pair_results", {})
                    st.write(
                        f"Scenarios: **{len(names)}** | Total pairs: **{out['summary']['total_pairs']}**"
                    )

                    # Summaries
                    am = out.get("accuracy_matrix")
                    if isinstance(am, np.ndarray) and am.size:
                        tri = np.triu_indices_from(am, k=1)
                        vals = am[tri]
                        if vals.size:
                            avg = float(np.mean(vals))
                            std = float(np.std(vals))
                            st.write(f"Avg pairwise acc: **{avg:.3f} ± {std:.3f}**")

                    if pair_results:
                        items = sorted(
                            (
                                {
                                    "pair": k.replace("_vs_", " vs "),
                                    "accuracy": v.get("accuracy", 0.0),
                                }
                                for k, v in pair_results.items()
                            ),
                            key=lambda x: x["accuracy"],
                            reverse=True,
                        )
                        top_n = items[: min(3, len(items))]
                        bot_n = items[-min(3, len(items)) :] if len(items) > 1 else []
                        if top_n:
                            st.markdown("**Top pairs**")
                            for it in top_n:
                                st.write(f"• {it['pair']} — acc={it['accuracy']:.3f}")
                        if bot_n:
                            st.markdown("**Challenging pairs**")
                            for it in bot_n:
                                st.write(f"• {it['pair']} — acc={it['accuracy']:.3f}")

            except Exception as e:
                st.error(f"Failed: {e}")

    def _render_group_vs_group(self, df: pd.DataFrame, clf: ScenarioClassifier, params: dict, dataset_root: str):
        st.markdown("### Group vs Group")
        unique_labels = self.scenario_manager.get_unique_labels(df)
        if len(unique_labels) < 2:
            st.error("Assign at least 2 distinct group labels in Scenarios panel.")
            return

        groups_info = self._analyze_groups(df)
        valid_groups = [g for g, c in groups_info.items() if c >= 2]
        if len(valid_groups) < 2:
            st.error("Need at least two groups with 2+ scenarios each.")
            return

        c1, c2 = st.columns(2)
        with c1:
            group_a = st.selectbox("Group A", options=valid_groups, index=0)
        with c2:
            group_b = st.selectbox(
                "Group B", options=[g for g in valid_groups if g != group_a], index=0
            )

        scen_a = self._get_group_scenarios(df, group_a)
        scen_b = self._get_group_scenarios(df, group_b)
        st.info(
            f"{group_a}: {len(scen_a)} scenarios | {group_b}: {len(scen_b)} scenarios"
        )

        c3, c4 = st.columns(2)
        with c3:
            max_samples = st.number_input(
                "Max samples per scenario", min_value=10, max_value=1000, value=100
            )
        with c4:
            balance = st.checkbox("Balance groups", value=True)

        if st.button("🚀 Run Group vs Group", type="primary"):
            try:
                out = clf.run_group_vs_group(
                    scenarios_a=scen_a,
                    scenarios_b=scen_b,
                    label_a=group_a,
                    label_b=group_b,
                    params={
                        **params,
                        "max_samples_per_scenario": int(max_samples),
                        "balance_groups": bool(balance),
                    },
                    dataset_root=dataset_root,
                )
                metrics: Dict[str, Any] = out["metrics"]

                # ---- Cache for Visualization panel ----
                st.session_state[SK_CLASSIFICATION_RESULTS] = {
                    **metrics,
                    "label_names": metrics.get("label_names", [group_a, group_b]),
                    "scenario_names": [group_a, group_b],
                }
                st.session_state[SK_LAST_MODEL_INFO] = clf.get_model_info()
                st.session_state[SK_CLASSIFICATION_ARTIFACTS] = {
                    "feature_names": out.get("feature_names", []),
                    "mode": "group_vs_group",
                }

                st.success(
                    f"Done. Test accuracy: {metrics.get('test_accuracy', 0):.3f}"
                )

                # ---- Compact Details expander ----
                with st.expander("Details"):
                    st.caption(
                        "Quick snapshot. Full confusion matrix and feature insights in Visualization."
                    )
                    st.write(
                        f"Labels: **{', '.join(metrics.get('label_names', [group_a, group_b]))}** | "
                        f"CV mean±std: **{metrics.get('cv_mean', 0):.3f}±{metrics.get('cv_std', 0):.3f}**"
                    )
                    st.write(
                        f"Train/Test samples: **{metrics.get('total_train', 0)}** / **{metrics.get('total_test', 0)}**"
                    )
            except Exception as e:
                st.error(f"Failed: {e}")

    # ----------------- Model management -----------------
    def _render_model_management(self, clf: ScenarioClassifier, dataset_root: str):
        st.markdown("---")
        st.markdown("### Model Management")
        info = clf.get_model_info()
        if clf.is_trained():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.success(
                    f"✅ Trained model: {info.get('model_type', '—').upper()} ({info.get('feature_type', '—')})"
                )
                st.caption(
                    f"Accuracy: {info.get('metrics', {}).get('test_accuracy', 0):.3f} | Mode: {info.get('mode', '—')}"
                )
            with c2:
                if st.button("💾 Save to dataset root"):
                    try:
                        path = clf.save_model(path=None, dataset_root=dataset_root)
                        st.success(f"Saved: {path}")
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            with c3:
                try:
                    fname, blob = clf.download_model()
                    st.download_button(
                        "📥 Download Model",
                        data=blob,
                        file_name=fname,
                        mime="application/octet-stream",
                    )
                except Exception as e:
                    st.error(f"Download unavailable: {e}")
        else:
            st.info("No trained model in session.")

        st.markdown("#### Load Existing Model File")
        up = st.file_uploader("Upload model file", type=["joblib", "pkl"])
        if up is not None:
            try:
                new_clf = ScenarioClassifier.load_model(file_bytes=up.read())
                st.session_state[SK_CLASSIFIER_OBJ] = new_clf
                info2 = new_clf.get_model_info()
                st.success(
                    f"Loaded model: {info2.get('model_type', '—').upper()} ({info2.get('feature_type', '—')})"
                )
            except Exception as e:
                st.error(f"Load failed: {e}")

    # ----------------- group helpers -----------------
    def _analyze_groups(self, df: pd.DataFrame):
        groups: Dict[str, int] = {}
        for _, row in df.iterrows():
            labels = [l.strip() for l in str(row.get('label', '')).split(',') if l.strip()]
            for l in labels:
                groups[l] = groups.get(l, 0) + 1
        return groups

    def _get_group_scenarios(self, df: pd.DataFrame, group_label: str):
        out: List[Tuple[str, str]] = []
        for _, row in df.iterrows():
            labels = [l.strip() for l in str(row.get('label', '')).split(',') if l.strip()]
            if group_label in labels:
                out.append((row['path'], row['scenario']))
        return out
