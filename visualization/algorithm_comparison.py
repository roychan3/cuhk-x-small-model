"""Algorithm-comparison page for the CUHK-X dashboard.

Reads ``artifacts/*/validation.json`` (the same files ``modeling/compare.py``
uses) without importing scikit-learn.  Automatically surfaces every registered
algorithm that has been trained — add a new file under ``modeling/algorithms/``
and re-run ``python -m modeling.train --algorithm <name>`` and it appears with
no dashboard code change.

The leaderboard columns and the download CSV come from
``visualization/comparison_format.py``, which ``modeling/compare.py`` shares,
so the page and the CLI cannot drift.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from visualization.comparison_format import COMPARISON_FIELDS, assign_labels, comparison_row

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = REPOSITORY_ROOT / "artifacts"
METRIC_CHOICES = ("macro_f1", "accuracy", "balanced_accuracy")


# ---------------------------------------------------------------------------
# Discovery / parsing — no sklearn dependency
# ---------------------------------------------------------------------------

def discover_validation_reports(artifacts_root: Path | str | None = None) -> list[Path]:
    root = Path(artifacts_root).expanduser() if artifacts_root is not None else DEFAULT_ARTIFACTS_ROOT
    if not root.is_dir():
        return []
    reports = sorted(root.glob("*/validation.json"))
    # Fallback: also accept flat layout (e.g. artifacts/validation.json)
    if not reports and (root / "validation.json").is_file():
        reports = [root / "validation.json"]
    return reports


def _row(path: Path, report: dict[str, Any]) -> dict[str, object]:
    """The shared comparison columns plus the extras only the dashboard shows."""

    return {
        **comparison_row(path, report),
        "display_name": report.get("algorithm_display_name") or report.get("algorithm") or path.parent.name,
        "parameters_dict": report.get("selected_parameters", {}),
        "selection_metric": report.get("selection_metric"),
        "test_clips": report.get("test_clips"),
        # `--skip-validation` runs have no `aggregate_metrics`, so the three
        # metric columns are already None; flag them so they can be filtered.
        "validation_skipped": bool(report.get("validation_skipped")),
    }


def display_columns(sort_metric: str) -> list[str]:
    """Leaderboard column order: the shared CSV fields plus ``display_name``.

    ``sort_metric`` is surfaced first for scanability but is already one of the
    three fixed metric columns, so it must be deduplicated — otherwise
    ``DataFrame[cols]`` raises ``ValueError: Duplicate column names found``.
    """

    columns = ["label", "algorithm", "display_name", "artifact_name", "parameters", sort_metric]
    columns += [field for field in COMPARISON_FIELDS if field not in columns]
    return columns


@st.cache_data(show_spinner=False)
def cached_validation_reports(report_paths: tuple[str, ...]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for raw in report_paths:
        path = Path(raw)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reports.append({"_path": str(path), "_error": str(exc)})
            continue
        report["_path"] = str(path)
        reports.append(report)
    return reports


def _class_labels(report: dict[str, Any]) -> list[str]:
    # Prefer mapping in Training/class_mapping.csv, fall back to 0..n-1
    try:
        mapping_path = REPOSITORY_ROOT / "Training" / "class_mapping.csv"
        if mapping_path.is_file():
            with mapping_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                id_to_name = {int(r["action_id"]): r["action_name"] for r in reader}
            n = len(report.get("confusion_matrix", []))
            if n and len(id_to_name) >= n:
                return [id_to_name.get(i, str(i)) for i in range(n)]
    except Exception:
        pass
    n = len(report.get("confusion_matrix", []))
    if n:
        return [str(i) for i in range(n)]
    return [str(i) for i in range(40)]


def _confusion_figure(
    matrix: list[list[int]],
    labels: list[str],
    title: str,
    *,
    row_normalize: bool = False,
    colorscale: str = "Blues",
    reversescale: bool = False,
) -> go.Figure:
    import numpy as np  # local import — numpy is already in modeling deps but not required here

    arr = np.asarray(matrix, dtype=float)
    if row_normalize:
        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        arr = arr / row_sums
        hovertemplate = "true=%{y}<br>pred=%{x}<br>recall=%{z:.3f}<extra></extra>"
        colorbar_title = "recall"
    else:
        hovertemplate = "true=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>"
        colorbar_title = "count"

    # Truncate labels for display if long (action names include prefix)
    display_labels = [lbl.split("_", 1)[-1].replace("_", " ") if "_" in lbl else lbl for lbl in labels]

    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=display_labels,
            y=display_labels,
            colorscale=colorscale,
            reversescale=reversescale,
            colorbar_title=colorbar_title,
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        height=520,
        xaxis={"tickangle": -55},
        yaxis={"autorange": "reversed"},
    )
    return fig


def _delta_figure(
    matrix_a: list[list[int]],
    matrix_b: list[list[int]],
    labels: list[str],
    title: str,
) -> go.Figure:
    import numpy as np

    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    # Normalize per-row to compare recall deltas fairly
    def _row_norm(m: np.ndarray) -> np.ndarray:
        s = m.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return m / s

    delta = _row_norm(a) - _row_norm(b)
    display_labels = [lbl.split("_", 1)[-1].replace("_", " ") if "_" in lbl else lbl for lbl in labels]
    # Symmetric diverging scale around 0
    vmax = float(max(0.05, abs(delta).max())) if delta.size else 0.05
    fig = go.Figure(
        data=go.Heatmap(
            z=delta,
            x=display_labels,
            y=display_labels,
            colorscale="RdBu",
            reversescale=True,
            zmin=-vmax,
            zmax=vmax,
            zmid=0,
            colorbar_title="Δ recall (A−B)",
            hovertemplate="true=%{y}<br>pred=%{x}<br>Δ=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        height=520,
        xaxis={"tickangle": -55},
        yaxis={"autorange": "reversed"},
    )
    return fig


def _fold_metric_series(report: dict[str, Any], metric: str) -> pd.DataFrame:
    """Return per-fold metric for the selected parameters, one row per fold."""
    selected = report.get("selected_parameters")
    rows: list[dict[str, object]] = []
    for fold in report.get("folds", []):
        fold_idx = fold.get("fold")
        best = None
        for cand in fold.get("candidates", []):
            if cand.get("parameters") == selected:
                best = cand
                break
        # Fallback: pick candidate with same params as aggregate best
        if best is None and fold.get("candidates"):
            best = fold["candidates"][0]
        if best is None:
            continue
        metrics = best.get("metrics", {})
        rows.append({"fold": int(fold_idx) if fold_idx is not None else len(rows) + 1, metric: metrics.get(metric)})
    return pd.DataFrame(rows)


def _leaderboard_csv(rows: list[dict[str, object]]) -> str:
    """Render the download CSV; ``COMPARISON_FIELDS`` keeps it equal to the CLI's."""

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COMPARISON_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in COMPARISON_FIELDS})
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

def render_algorithm_comparison() -> None:
    st.header("Algorithm comparison")
    st.caption(
        "Compare every trained algorithm on the shared multimodal representation "
        "built from Depth Color, IR, IMU, and Skeleton — PCA-reduced base blocks "
        "plus pass-through engineered features. Widths depend on the feature "
        "configuration; each report records its own under `raw_feature_dimensions`. "
        "Each run writes `artifacts/<algorithm>/validation.json`; "
        "this page and `python -m modeling.compare` read the same files. "
        "Add a new algorithm under `modeling/algorithms/` and rerun "
        "`python -m modeling.train --algorithm <name>` — it appears here with no dashboard change."
    )

    artifacts_root = DEFAULT_ARTIFACTS_ROOT
    report_paths = discover_validation_reports(artifacts_root)

    # Allow user to point at a different artifacts checkout (useful in Docker)
    with st.expander("Artifacts location", expanded=False):
        st.code(str(artifacts_root), language="text")
        st.caption(f"Default: `artifacts/*/validation.json` relative to the repository root. Discovered {len(report_paths)} report(s).")
        override = st.text_input("Override artifacts root (optional)", value="", placeholder=str(artifacts_root))
        if override.strip():
            alt_paths = discover_validation_reports(Path(override.strip()).expanduser())
            if alt_paths:
                report_paths = alt_paths
                st.success(f"Using override — found {len(report_paths)} report(s).")
            else:
                st.warning(f"No `*/validation.json` under {override.strip()} — showing default.")

    if not report_paths:
        st.info(
            "No validation reports found. Train an algorithm first:\n\n"
            "`python -m modeling.train --algorithm logistic_regression --dataset-root /path/to/small-model --n-jobs 8`\n\n"
            "Reports appear as `artifacts/logreg/validation.json` (and one per future algorithm). "
            "Run `python -m modeling.compare --help` for the CLI equivalent."
        )
        st.stop()

    reports = cached_validation_reports(tuple(str(p) for p in report_paths))

    # Surface parse errors inline
    parse_errors = [r for r in reports if "_error" in r]
    if parse_errors:
        for err in parse_errors:
            st.warning(f"Could not read {err['_path']}: {err['_error']}")

    valid_reports: list[dict[str, Any]] = [r for r in reports if "_error" not in r]
    if not valid_reports:
        st.error("All discovered reports failed to parse.")
        st.stop()

    # Rows carry the normalized identity (`label`); the report they came from is
    # looked up by that same label everywhere below, so a report missing the
    # `algorithm` key can never fall out of the figures while staying in the table.
    rows = assign_labels([_row(Path(r["_path"]), r) for r in valid_reports])
    report_by_label: dict[str, dict[str, Any]] = {
        str(row["label"]): report for row, report in zip(rows, valid_reports)
    }

    # --- Controls ---
    all_algos = sorted({str(r["algorithm"]) for r in rows})
    # Stable default sort
    default_metric = "macro_f1"

    ctrl_left, ctrl_mid, ctrl_right = st.columns((1.1, 1, 1))
    with ctrl_left:
        sort_metric = st.selectbox("Sort by", METRIC_CHOICES, index=METRIC_CHOICES.index(default_metric))
    with ctrl_mid:
        selected_algos = st.multiselect("Algorithms", all_algos, default=all_algos)
    with ctrl_right:
        include_skipped = st.checkbox("Include skipped-validation runs", value=False)

    if not include_skipped:
        rows = [r for r in rows if not r.get("validation_skipped")]

    if selected_algos:
        rows = [r for r in rows if r["algorithm"] in selected_algos]

    if not rows:
        st.warning("No reports match the current filters.")
        st.stop()

    # Sort descending by chosen metric (None last)
    def _sort_key(r: dict[str, object]) -> float:
        v = r.get(sort_metric)
        return float(v) if isinstance(v, (int, float)) else -1.0

    rows.sort(key=_sort_key, reverse=True)
    # Reports aligned with the sorted rows, keyed by the unique label
    ordered: list[tuple[str, dict[str, Any]]] = [(str(r["label"]), report_by_label[str(r["label"])]) for r in rows]

    # --- Leaderboard ---
    st.subheader("Leaderboard")
    leaderboard_df = pd.DataFrame(rows)
    display_cols = display_columns(sort_metric)
    # Ensure all expected columns exist
    for col in display_cols:
        if col not in leaderboard_df.columns:
            leaderboard_df[col] = None
    # Highlight best
    best_value = _sort_key(rows[0]) if rows else None
    st.dataframe(
        leaderboard_df[display_cols],
        hide_index=True,
        width="stretch",
        column_config={
            "label": st.column_config.TextColumn("Run"),
            "algorithm": st.column_config.TextColumn("Algorithm"),
            "display_name": st.column_config.TextColumn("Display name"),
            "artifact_name": st.column_config.TextColumn("Artifact dir"),
            "parameters": st.column_config.TextColumn("Parameters", width="medium"),
            "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
            "macro_f1": st.column_config.NumberColumn("Macro F1", format="%.4f"),
            "balanced_accuracy": st.column_config.NumberColumn("Bal. acc.", format="%.4f"),
            "training_clips": st.column_config.NumberColumn("Training clips"),
            "report": st.column_config.TextColumn("Report"),
        },
    )
    if best_value is not None and best_value >= 0:
        st.caption(f"Best by **{sort_metric}**: `{rows[0]['label']}` ({best_value:.4f}). Sorted descending; `None` (skipped validation) last.")

    # Version-mismatch warnings
    for label, rep in ordered:
        if not rep.get("library_versions"):
            st.warning(f"`{label}` report records no `library_versions` — retrain to record them.")
        # Detailed mismatch is shown in the metadata expander; keep banner quiet unless user inspects

    # CSV download — same columns as CLI
    csv_text = _leaderboard_csv(rows)
    st.download_button(
        "Download comparison CSV",
        data=csv_text,
        file_name="algorithm_comparison.csv",
        mime="text/csv",
    )
    st.caption("CSV columns match `python -m modeling.compare --output comparison.csv`.")

    # --- Metrics bar chart ---
    st.subheader("Metrics overview")
    metric_view = st.radio("Metric view", ("Selected metric", "All three metrics"), horizontal=True)
    if metric_view == "Selected metric":
        plot_df = pd.DataFrame([{"run": r["label"], sort_metric: r.get(sort_metric)} for r in rows])
        plot_df = plot_df.dropna(subset=[sort_metric])
        if not plot_df.empty:
            fig = px.bar(plot_df, x="run", y=sort_metric, text=sort_metric, color="run")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(height=420, showlegend=False, yaxis_title=sort_metric, xaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(f"No `{sort_metric}` values to plot (all skipped validation).")
    else:
        long_rows: list[dict[str, object]] = []
        for r in rows:
            for m in METRIC_CHOICES:
                v = r.get(m)
                if isinstance(v, (int, float)):
                    long_rows.append({"run": r["label"], "metric": m, "value": float(v)})
        if long_rows:
            long_df = pd.DataFrame(long_rows)
            fig = px.bar(long_df, x="run", y="value", color="metric", barmode="group", text="value")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=440, yaxis_title="score", xaxis_title=None, legend_title=None)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No metric values to plot.")

    # --- Confusion matrices ---
    st.subheader("Confusion matrices")
    # Find reports that actually have a confusion matrix
    with_cm = [(label, r) for label, r in ordered if isinstance(r.get("confusion_matrix"), list) and len(r["confusion_matrix"]) > 0]
    if not with_cm:
        st.info("No confusion matrix found in the selected reports (skipped validation or older artifact).")
    else:
        cm_by_label = dict(with_cm)
        cm_labels = [label for label, _ in with_cm]
        # Default selection: best vs second-best if 2+ runs
        default_b = cm_labels[1] if len(cm_labels) > 1 else cm_labels[0]
        c1, c2, c3 = st.columns((1, 1, 1))
        with c1:
            algo_a = st.selectbox("Matrix A", cm_labels, index=0, key="cm_a")
        with c2:
            algo_b = st.selectbox("Matrix B", cm_labels, index=cm_labels.index(default_b), key="cm_b")
        with c3:
            row_norm = st.checkbox("Row-normalize (per-true-class recall)", value=False)

        rep_a = cm_by_label[algo_a]
        labels_a = _class_labels(rep_a)

        if algo_a == algo_b or len(with_cm) == 1:
            title = f"{algo_a} — {'recall' if row_norm else 'counts'}"
            fig = _confusion_figure(rep_a["confusion_matrix"], labels_a, title, row_normalize=row_norm)
            st.plotly_chart(fig, width="stretch")
            st.caption("Rows = true class, columns = predicted. 40 classes (0–39).")
        else:
            rep_b = cm_by_label[algo_b]
            # Show A and B side-by-side then delta
            col_a, col_b = st.columns(2)
            with col_a:
                fig_a = _confusion_figure(rep_a["confusion_matrix"], labels_a, f"{algo_a} — {'recall' if row_norm else 'counts'}", row_normalize=row_norm)
                st.plotly_chart(fig_a, width="stretch")
            with col_b:
                # Use same label set as A for alignment (both 40 classes)
                fig_b = _confusion_figure(rep_b["confusion_matrix"], labels_a, f"{algo_b} — {'recall' if row_norm else 'counts'}", row_normalize=row_norm)
                st.plotly_chart(fig_b, width="stretch")

            st.markdown("**Delta (A − B) — row-normalized recall**")
            fig_delta = _delta_figure(rep_a["confusion_matrix"], rep_b["confusion_matrix"], labels_a, f"Δ recall: {algo_a} − {algo_b}")
            st.plotly_chart(fig_delta, width="stretch")
            st.caption("Positive (red) = A better recall for that true→pred cell; negative (blue) = B better. Computed on row-normalized matrices so class frequency does not dominate.")

        # Per-class summary for quick scanning (optional table)
        with st.expander("Per-class F1 (derived from confusion matrix)", expanded=False):
            try:
                import numpy as np

                def _per_class_f1(cm: list[list[int]]) -> pd.DataFrame:
                    arr = np.asarray(cm, dtype=float)
                    n = arr.shape[0]
                    precisions: list[float] = []
                    recalls: list[float] = []
                    f1s: list[float] = []
                    for i in range(n):
                        tp = arr[i, i]
                        fp = arr[:, i].sum() - tp
                        fn = arr[i, :].sum() - tp
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                        precisions.append(prec)
                        recalls.append(rec)
                        f1s.append(f1)
                    return pd.DataFrame({"class_id": range(n), "precision": precisions, "recall": recalls, "f1": f1s})

                df_a = _per_class_f1(rep_a["confusion_matrix"])
                df_a["run"] = algo_a
                if algo_a != algo_b:
                    df_b = _per_class_f1(rep_b["confusion_matrix"])
                    df_b["run"] = algo_b
                    combined = pd.concat([df_a, df_b], ignore_index=True)
                    fig = px.bar(combined, x="class_id", y="f1", color="run", barmode="group")
                    fig.update_layout(height=360, xaxis_title="Action ID", yaxis_title="F1")
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(combined.sort_values(["class_id", "run"]), hide_index=True, width="stretch")
                else:
                    fig = px.bar(df_a, x="class_id", y="f1")
                    fig.update_layout(height=360, xaxis_title="Action ID", yaxis_title="F1")
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(df_a, hide_index=True, width="stretch")
            except Exception as exc:
                st.warning(f"Could not compute per-class F1: {exc}")

    # --- Folds ---
    st.subheader("Cross-validation folds")
    fold_candidates = [(label, r) for label, r in ordered if isinstance(r.get("folds"), list) and len(r["folds"]) > 0]
    if not fold_candidates:
        st.info("No fold details in the selected reports (skipped validation or older artifact).")
    else:
        fold_by_label = dict(fold_candidates)
        fold_algo = st.selectbox("Inspect folds for", list(fold_by_label), key="fold_algo")
        rep_fold = fold_by_label[fold_algo]
        metric_fold = st.selectbox("Fold metric", METRIC_CHOICES, index=METRIC_CHOICES.index(sort_metric), key="fold_metric")
        df_folds = _fold_metric_series(rep_fold, metric_fold)
        if not df_folds.empty and df_folds[metric_fold].notna().any():
            fig = px.line(df_folds, x="fold", y=metric_fold, markers=True)
            fig.update_layout(height=360, xaxis_title="Fold", yaxis_title=metric_fold)
            # Integer ticks
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(f"No `{metric_fold}` values per fold for `{fold_algo}`.")

        with st.expander("Fold details", expanded=False):
            # Show raw folds table
            flat: list[dict[str, object]] = []
            for fold in rep_fold.get("folds", []):
                for cand in fold.get("candidates", []):
                    flat.append(
                        {
                            "fold": fold.get("fold"),
                            "validation_clips": fold.get("validation_clips"),
                            "parameters": json.dumps(cand.get("parameters"), sort_keys=True),
                            metric_fold: cand.get("metrics", {}).get(metric_fold),
                            "accuracy": cand.get("metrics", {}).get("accuracy"),
                            "macro_f1": cand.get("metrics", {}).get("macro_f1"),
                        }
                    )
            if flat:
                st.dataframe(pd.DataFrame(flat), hide_index=True, width="stretch")
            # Candidate aggregate table
            agg_rows = rep_fold.get("aggregate_metrics", [])
            if agg_rows:
                agg_df = pd.DataFrame(
                    [
                        {
                            "parameters": json.dumps(r.get("parameters"), sort_keys=True),
                            "accuracy": r.get("metrics", {}).get("accuracy"),
                            "macro_f1": r.get("metrics", {}).get("macro_f1"),
                            "balanced_accuracy": r.get("metrics", {}).get("balanced_accuracy"),
                        }
                        for r in agg_rows
                    ]
                )
                st.markdown("**Aggregate (cross-validated) metrics per candidate**")
                st.dataframe(agg_df, hide_index=True, width="stretch")

    # --- Run metadata ---
    with st.expander("Run metadata & health", expanded=False):
        meta_by_label = dict(ordered)
        meta_algo = st.selectbox("Metadata for", list(meta_by_label), key="meta_algo")
        rep_meta = meta_by_label[meta_algo]
        # Use two columns for readability
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Selected parameters**")
            st.json(rep_meta.get("selected_parameters", {}))
            st.markdown("**Library versions**")
            st.json(rep_meta.get("library_versions", {}))
            versions = rep_meta.get("library_versions", {})
            if versions:
                # Compare to running versions if modeling is available
                try:
                    from modeling.model import library_versions as _current_versions

                    current = _current_versions()
                    mismatched = {k: (versions.get(k), current.get(k)) for k in current if versions.get(k) != current.get(k)}
                    if mismatched:
                        detail = "; ".join(f"{k} {v[0]} → {v[1]}" for k, v in sorted(mismatched.items()))
                        st.warning(f"Artifact was written with different versions ({detail}). Retrain to be certain.")
                    else:
                        st.success("Artifact library versions match the running environment.")
                except Exception:
                    pass
            else:
                st.warning("No `library_versions` recorded in this artifact.")
        with m2:
            st.markdown("**Dataset & filtering**")
            st.json(
                {
                    "training_clips": rep_meta.get("training_clips"),
                    "test_clips": rep_meta.get("test_clips"),
                    "dataset_root": rep_meta.get("dataset_root"),
                    "selection_metric": rep_meta.get("selection_metric"),
                    "validation_skipped": rep_meta.get("validation_skipped", False),
                }
            )
            st.markdown("**Test feature health**")
            st.json(rep_meta.get("test_feature_health", {}))
            if rep_meta.get("test_feature_health"):
                st.caption("`partial`/`all_missing` rows are mean-imputed from training statistics within each fold.")
            if rep_meta.get("class_counts"):
                # Show as small bar
                cc = rep_meta["class_counts"]
                if isinstance(cc, dict):
                    cc_df = pd.DataFrame([{"action_id": int(k), "clips": int(v)} for k, v in cc.items()]).sort_values("action_id")
                    fig = px.bar(cc_df, x="action_id", y="clips")
                    fig.update_layout(height=260, xaxis_title="Action ID", yaxis_title="clips", showlegend=False)
                    st.plotly_chart(fig, width="stretch")

        st.markdown("**Full report JSON**")
        # Don't dump huge confusion_matrix inline by default — offer toggle
        show_full = st.checkbox("Show full JSON including confusion matrix", value=False, key="show_full_json")
        if show_full:
            st.json(rep_meta)
        else:
            trimmed = {k: v for k, v in rep_meta.items() if k != "confusion_matrix"}
            trimmed["confusion_matrix"] = f"<{len(rep_meta.get('confusion_matrix', []))}×{len(rep_meta.get('confusion_matrix', [[]])[0]) if rep_meta.get('confusion_matrix') else 0} omitted>"
            st.json(trimmed)
