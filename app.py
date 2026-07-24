# app.py
"""
CIMT Volumetric 3D Explorer — Gradio Application
=================================================
2-level cascade (System -> Sub-system) narrows 448 ROIs to a small
checklist. The user manually ticks the ROIs to render.
Extra indices can be added via Advanced.

Layout: Minimalist, modern. White background. 3D viewer dominates the
left; a compact control sidebar sits on the right. Clean hierarchy,
generous whitespace, single accent colour.

Deployment:
    Place alongside a `data/` folder containing:
      - CIMT_448ROIs_atlas.nii.gz
      - cimt_atlas_labels.csv
    Compatible with HuggingFace Spaces (Gradio SDK).

Usage:
    python app.py
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import to_hex
from skimage import measure

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cimt_explorer")


SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
ATLAS_FILENAME: str = "CIMT_448ROIs_atlas.nii.gz"
LABELS_FILENAME: str = "cimt_atlas_labels.csv"

MAX_ROIS_RENDER: int = 80
DEFAULT_ALPHA: float = 0.65
DEFAULT_CMAP: str = "plasma"
DEFAULT_LEGEND_MODE: str = "auto"
BRAIN_OPACITY: float = 0.08
BRAIN_COLOR: str = "#a8b0ba"
SUBSYSTEM_ALL: str = "All Sub-systems"
AUTO_FULL_LEGEND_THRESHOLD: int = 12

CMAP_CHOICES: List[str] = [
    "plasma", "viridis", "inferno", "coolwarm", "tab20", "Set1", "Set2",
]

LEGEND_MODE_CHOICES: List[str] = [
    "auto", "full", "region_full_name", "roi_name",
]

# ---------------------------------------------------------------------------
# Design Tokens
# ---------------------------------------------------------------------------

_CLR_BG: str = "#ffffff"
_CLR_SURFACE: str = "#ffffff"
_CLR_BORDER: str = "#e8eaed"
_CLR_BORDER_FOCUS: str = "#2c3e50"
_CLR_ACCENT: str = "#2c3e50"
_CLR_ACCENT_HOVER: str = "#3d5266"
_CLR_TEXT: str = "#1a1a2e"
_CLR_TEXT_SECONDARY: str = "#6b7280"
_CLR_TEXT_MUTED: str = "#9ca3af"
_CLR_SCENE_BG: str = "#f8f9fa"
_CLR_HOVER_BG: str = "#f3f4f6"

# ---------------------------------------------------------------------------
# Application CSS
# ---------------------------------------------------------------------------

APP_CSS: str = """
/* ── Fonts ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ─────────────────────────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #ffffff !important;
    max-width: 1520px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

/* ── Header ─────────────────────────────────────────────────────────────── */
.app-header {
    padding: 28px 0 16px;
    margin-bottom: 0;
}
.app-header h1 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
    letter-spacing: -0.5px;
    margin: 0 !important;
}
.app-header p {
    font-size: 13px !important;
    color: #6b7280 !important;
    font-weight: 400;
    margin: 4px 0 0 0 !important;
}

/* ── Main Layout ────────────────────────────────────────────────────────── */
.main-layout {
    gap: 20px !important;
    align-items: stretch;
}

/* ── Viewer ─────────────────────────────────────────────────────────────── */
.viewer-panel {
    border: 1px solid #e8eaed;
    border-radius: 12px;
    overflow: hidden;
    background: #f8f9fa;
    display: flex;
    flex-direction: column;
}
.viewer-panel .plot-container {
    flex: 1;
    min-height: 0;
}
.viewer-panel .plot-container > div {
    height: 100% !important;
}
.viewer-panel .js-plotly-plot,
.viewer-panel .plotly-graph-div {
    height: 100% !important;
    width: 100% !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
.sidebar {
    display: flex;
    flex-direction: column;
    gap: 0;
}

/* ── Section Blocks ─────────────────────────────────────────────────────── */
.section-block {
    padding: 20px 0;
    border-bottom: 1px solid #e8eaed;
}
.section-block:last-child {
    border-bottom: none;
}
.section-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    color: #9ca3af;
    margin-bottom: 14px;
}

/* ── Dropdowns ──────────────────────────────────────────────────────────── */
.gradio-container select,
.gradio-container .wrap.svelte-1ipelgc {
    background: #ffffff !important;
    border: 1px solid #e8eaed !important;
    border-radius: 8px !important;
    color: #1a1a2e !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.gradio-container select:focus,
.gradio-container select:hover {
    border-color: #2c3e50 !important;
    box-shadow: 0 0 0 3px rgba(44, 62, 80, 0.06) !important;
    outline: none;
}
.gradio-container label > span {
    color: #6b7280 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.gradio-container button.primary {
    background: #2c3e50 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 24px !important;
    letter-spacing: 0.2px;
    transition: all 0.2s ease;
    width: 100%;
}
.gradio-container button.primary:hover {
    background: #3d5266 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(44, 62, 80, 0.15);
}
.gradio-container button.primary:active {
    transform: translateY(0);
    box-shadow: none;
}

/* ── ROI Checklist (compact: ~4 visible rows) ───────────────────────────── */
.roi-checklist > div {
    max-height: 120px;
    overflow-y: auto;
    border: 1px solid #e8eaed;
    border-radius: 8px;
    padding: 6px 10px;
    background: #ffffff;
}
.roi-checklist label {
    font-size: 12px;
    color: #1a1a2e;
    padding: 4px 8px;
    border-radius: 5px;
    transition: background 0.1s ease;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.3;
}
.roi-checklist label:hover {
    background: #f3f4f6;
}
.roi-checklist input[type="checkbox"] {
    accent-color: #2c3e50;
    width: 14px;
    height: 14px;
    flex-shrink: 0;
}
.roi-checklist input[type="checkbox"]:checked + span {
    color: #2c3e50;
    font-weight: 600;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
.roi-checklist > div::-webkit-scrollbar {
    width: 4px;
}
.roi-checklist > div::-webkit-scrollbar-track {
    background: transparent;
}
.roi-checklist > div::-webkit-scrollbar-thumb {
    background: #d1d5db;
    border-radius: 2px;
}
.roi-checklist > div::-webkit-scrollbar-thumb:hover {
    background: #9ca3af;
}

/* ── Status ─────────────────────────────────────────────────────────────── */
.status-bar textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    background: #f8f9fa !important;
    border: 1px solid #e8eaed !important;
    border-radius: 6px !important;
    color: #6b7280 !important;
    padding: 8px 12px !important;
}

/* ── Accordion ──────────────────────────────────────────────────────────── */
.gradio-container .accordion {
    border: 1px solid #e8eaed !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}
.gradio-container .accordion .label-wrap span {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #9ca3af !important;
}

/* ── Sliders ────────────────────────────────────────────────────────────── */
.gradio-container input[type="range"] {
    accent-color: #2c3e50;
}

/* ── Checkboxes (general) ───────────────────────────────────────────────── */
.gradio-container input[type="checkbox"] {
    accent-color: #2c3e50;
}

/* ── Footer ─────────────────────────────────────────────────────────────── */
.app-footer {
    font-size: 11px;
    color: #9ca3af;
    text-align: center;
    padding: 20px 0 28px;
    margin-top: 8px;
    letter-spacing: 0.3px;
}

/* ── Plotly Overrides ───────────────────────────────────────────────────── */
.js-plotly-plot .plotly .modebar {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 6px;
    padding: 4px;
    border: 1px solid #e8eaed;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}
.js-plotly-plot .plotly .modebar-btn path {
    fill: #6b7280 !important;
}
.js-plotly-plot .plotly .modebar-btn:hover path {
    fill: #2c3e50 !important;
}

/* ── Misc ───────────────────────────────────────────────────────────────── */
.gradio-container .prose {
    color: #1a1a2e !important;
}
.gradio-container .info-text,
.gradio-container .info {
    color: #9ca3af !important;
    font-size: 11px !important;
}
"""

# ---------------------------------------------------------------------------
# Module-Level Singletons
# ---------------------------------------------------------------------------

_ATLAS_IMG: Optional[nib.Nifti1Image] = None
_ATLAS_DATA: Optional[np.ndarray] = None
_LABELS_DF: Optional[pd.DataFrame] = None
_BRAIN_VERTS: Optional[np.ndarray] = None
_BRAIN_FACES: Optional[np.ndarray] = None
_LABEL_TO_INDEX: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Display Label Builder
# ---------------------------------------------------------------------------

def build_display_labels(labels_df: pd.DataFrame) -> pd.Series:
    """Create unique, human-readable labels for the ROI checklist.

    Rules:
        1. Use region_full_name as the primary label.
        2. Append hemisphere abbreviation unless the full name already
           ends with '(Left)' or '(Right)' (e.g., cerebellar ROIs).
        3. If the resulting label is duplicated, append the ROI code.
        4. If still duplicated, append the atlas index.

    Args:
        labels_df: Atlas labels DataFrame with required columns.

    Returns:
        Series of unique display labels aligned with labels_df index.

    Raises:
        ValueError: If labels cannot be made unique.
    """
    full_name = labels_df["region_full_name"].fillna("Unknown").astype(str)
    roi_name = labels_df["roi_name"].fillna("unknown").astype(str)
    index_col = labels_df["index"].astype(int)
    hemisphere_abbr = labels_df["hemisphere"].fillna("?").str[0].str.upper()

    already_has_side = full_name.str.endswith(("(Left)", "(Right)"))
    base_label = full_name.where(
        already_has_side,
        full_name + " (" + hemisphere_abbr + ")",
    )

    counts = base_label.value_counts()
    is_duplicated = base_label.map(counts) > 1

    display_label = base_label.where(
        ~is_duplicated,
        base_label + " [" + roi_name + "]",
    )

    still_duplicated = display_label.duplicated(keep=False)
    display_label = display_label.where(
        ~still_duplicated,
        display_label + " [#" + index_col.astype(str) + "]",
    )

    if not display_label.is_unique:
        raise ValueError(
            "Display labels are not unique after all disambiguation steps."
        )

    return display_label


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_atlas_data() -> None:
    """Load NIfTI volume, labels CSV, and brain mesh into singletons."""
    global _ATLAS_IMG, _ATLAS_DATA, _LABELS_DF
    global _BRAIN_VERTS, _BRAIN_FACES, _LABEL_TO_INDEX

    atlas_path = DATA_DIR / ATLAS_FILENAME
    labels_path = DATA_DIR / LABELS_FILENAME

    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Atlas not found: {atlas_path}\n"
            f"Place '{ATLAS_FILENAME}' in '{DATA_DIR}/'."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels not found: {labels_path}\n"
            f"Place '{LABELS_FILENAME}' in '{DATA_DIR}/'."
        )

    logger.info("Loading NIfTI atlas...")
    _ATLAS_IMG = nib.load(str(atlas_path))
    _ATLAS_DATA = _ATLAS_IMG.get_fdata()

    logger.info("Loading labels CSV...")
    _LABELS_DF = pd.read_csv(labels_path)

    expected = np.arange(len(_LABELS_DF))
    actual = _LABELS_DF["index"].astype(int).to_numpy()
    if not np.array_equal(actual, expected):
        raise ValueError(
            "Atlas 'index' column must be contiguous starting at 0. "
            f"Got range [{actual.min()}, {actual.max()}] with {len(actual)} rows."
        )

    _LABELS_DF["display_label"] = build_display_labels(_LABELS_DF)
    _LABEL_TO_INDEX = dict(
        zip(_LABELS_DF["display_label"], _LABELS_DF["index"].astype(int))
    )

    _BRAIN_VERTS, _BRAIN_FACES = _load_brain_mesh()

    logger.info(
        "Ready: %d ROIs, %d systems, brain mesh=%s",
        len(_LABELS_DF),
        _LABELS_DF["functional_system"].nunique(),
        _BRAIN_VERTS is not None,
    )


def _load_brain_mesh() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load fsaverage5 pial surface. Non-fatal on failure."""
    try:
        from nilearn.datasets import load_fsaverage

        fsavg = load_fsaverage(mesh="fsaverage5")
        pial = fsavg.pial
        verts = np.vstack([
            pial.parts["left"].coordinates,
            pial.parts["right"].coordinates,
        ])
        faces = np.vstack([
            pial.parts["left"].faces,
            pial.parts["right"].faces + len(pial.parts["left"].coordinates),
        ])
        return verts, faces
    except Exception as exc:
        logger.warning("Brain mesh unavailable: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Cascade Logic
# ---------------------------------------------------------------------------

def get_systems() -> List[str]:
    """Sorted unique functional systems."""
    assert _LABELS_DF is not None
    return sorted(_LABELS_DF["functional_system"].dropna().unique().tolist())


def get_subsystems(system: Optional[str]) -> List[str]:
    """Sub-system options filtered by system."""
    assert _LABELS_DF is not None
    df = _LABELS_DF
    if system:
        df = df[df["functional_system"] == system]
    subs = sorted(df["sub_system"].dropna().unique().tolist())
    return [SUBSYSTEM_ALL] + subs


def get_filtered_roi_labels(
    system: Optional[str],
    subsystem: Optional[str],
) -> List[str]:
    """Return sorted display labels for ROIs matching the cascade filters."""
    assert _LABELS_DF is not None
    df = _LABELS_DF
    if system:
        df = df[df["functional_system"] == system]
    if subsystem and subsystem != SUBSYSTEM_ALL:
        df = df[df["sub_system"] == subsystem]
    return sorted(df["display_label"].tolist())


# ---------------------------------------------------------------------------
# Mesh Extraction
# ---------------------------------------------------------------------------

def extract_roi_mesh(label_value: int) -> Optional[Dict[str, np.ndarray]]:
    """Extract isosurface for one ROI using the cached volume data."""
    assert _ATLAS_DATA is not None and _ATLAS_IMG is not None

    mask = (_ATLAS_DATA == label_value).astype(np.float32)
    if not np.any(mask):
        return None
    try:
        verts, faces, normals, _ = measure.marching_cubes(mask, level=0.5)
    except Exception as exc:
        logger.warning("Marching cubes failed for label %d: %s", label_value, exc)
        return None

    verts_mni = nib.affines.apply_affine(_ATLAS_IMG.affine, verts)
    return {"vertices": verts_mni, "faces": faces, "normals": normals}


# ---------------------------------------------------------------------------
# Legend Label Resolution
# ---------------------------------------------------------------------------

def resolve_plot_label(
    row: pd.Series,
    legend_mode: str,
    n_traces: int,
) -> str:
    """Resolve the label shown in the Plotly legend."""
    mode = legend_mode

    if mode == "auto":
        mode = "full" if n_traces <= AUTO_FULL_LEGEND_THRESHOLD else "roi_name"

    if mode == "full":
        return str(row["display_label"])
    if mode == "region_full_name":
        return str(row["region_full_name"])
    return str(row["roi_name"])


# ---------------------------------------------------------------------------
# Figure Builder
# ---------------------------------------------------------------------------

def _apply_scene_layout(fig: go.Figure) -> go.Figure:
    """Apply consistent scene styling to a figure (responsive sizing)."""
    fig.update_layout(
        autosize=True,
        paper_bgcolor=_CLR_SCENE_BG,
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(
                eye=dict(x=0.0, y=1.8, z=0.4),
                up=dict(x=0.0, y=0.0, z=1.0),
                center=dict(x=0.0, y=0.0, z=0.0),
            ),
            bgcolor=_CLR_SCENE_BG,
            aspectmode="data",
        ),
        margin=dict(t=12, b=12, l=12, r=12),
    )
    return fig


def build_figure(
    selected_indices: List[int],
    legend_mode: str = DEFAULT_LEGEND_MODE,
    cmap: str = DEFAULT_CMAP,
    alpha: float = DEFAULT_ALPHA,
    show_brain: bool = True,
) -> go.Figure:
    """Construct Plotly 3D figure from ROI indices."""
    assert _LABELS_DF is not None

    if len(selected_indices) > MAX_ROIS_RENDER:
        logger.warning(
            "Capping ROIs: %d -> %d", len(selected_indices), MAX_ROIS_RENDER
        )
        selected_indices = selected_indices[:MAX_ROIS_RENDER]

    cmap_func = plt.get_cmap(cmap)
    fig = go.Figure()

    # Brain reference mesh.
    if show_brain and _BRAIN_VERTS is not None and _BRAIN_FACES is not None:
        fig.add_trace(go.Mesh3d(
            x=_BRAIN_VERTS[:, 0],
            y=_BRAIN_VERTS[:, 1],
            z=_BRAIN_VERTS[:, 2],
            i=_BRAIN_FACES[:, 0],
            j=_BRAIN_FACES[:, 1],
            k=_BRAIN_FACES[:, 2],
            color=BRAIN_COLOR,
            opacity=BRAIN_OPACITY,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Extract meshes.
    meshes: Dict[int, Dict[str, np.ndarray]] = {}
    display_names: Dict[int, str] = {}

    for idx in selected_indices:
        row = _LABELS_DF.iloc[idx]
        mesh = extract_roi_mesh(idx + 1)
        if mesh is not None:
            meshes[idx] = mesh
            display_names[idx] = str(row["display_label"])

    if not meshes:
        raise RuntimeError("No meshes extracted for selected ROIs.")

    # Add ROI traces.
    n_traces = len(meshes)
    for rank, (idx, mesh) in enumerate(meshes.items()):
        color = to_hex(cmap_func(rank / max(n_traces - 1, 1)))
        row = _LABELS_DF.iloc[idx]

        legend_label = resolve_plot_label(row, legend_mode, n_traces)

        hover_text = (
            f"<b>{row['display_label']}</b><br>"
            f"ROI code: {row['roi_name']}<br>"
            f"System: {row['functional_system']}<br>"
            f"Sub-system: {row['sub_system']}<br>"
            f"Hemisphere: {row['hemisphere']}<br>"
            f"Index: {idx}"
        )

        fig.add_trace(go.Mesh3d(
            x=mesh["vertices"][:, 0],
            y=mesh["vertices"][:, 1],
            z=mesh["vertices"][:, 2],
            i=mesh["faces"][:, 0],
            j=mesh["faces"][:, 1],
            k=mesh["faces"][:, 2],
            color=color,
            opacity=alpha,
            name=legend_label,
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=True,
        ))

    _apply_scene_layout(fig)

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor=_CLR_BORDER,
            borderwidth=1,
            font=dict(size=12, color=_CLR_TEXT, family="Inter, sans-serif"),
            title=dict(
                text="<b>Regions</b>",
                font=dict(size=13, color=_CLR_TEXT_SECONDARY),
            ),
            itemsizing="constant",
        ),
    )

    return fig


def build_initial_figure() -> go.Figure:
    """Build the initial brain-only figure shown on load."""
    fig = go.Figure()

    if _BRAIN_VERTS is not None and _BRAIN_FACES is not None:
        fig.add_trace(go.Mesh3d(
            x=_BRAIN_VERTS[:, 0],
            y=_BRAIN_VERTS[:, 1],
            z=_BRAIN_VERTS[:, 2],
            i=_BRAIN_FACES[:, 0],
            j=_BRAIN_FACES[:, 1],
            k=_BRAIN_FACES[:, 2],
            color=BRAIN_COLOR,
            opacity=BRAIN_OPACITY,
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        width=860,
        height=780,
        paper_bgcolor=_CLR_SCENE_BG,
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(
                eye=dict(x=0.0, y=1.8, z=0.4),
                up=dict(x=0.0, y=0.0, z=1.0),
                center=dict(x=0.0, y=0.0, z=0.0),
            ),
            bgcolor=_CLR_SCENE_BG,
            aspectmode="data",
        ),
        margin=dict(t=12, b=12, l=12, r=12),
    )

    return fig


# ---------------------------------------------------------------------------
# Gradio Callbacks
# ---------------------------------------------------------------------------

def on_system_change(
    system: Optional[str],
) -> Tuple[gr.Dropdown, gr.CheckboxGroup]:
    """Cascade: System changed -> update Sub-system + checklist."""
    sys_val = system if system else None
    subs = get_subsystems(sys_val)
    roi_labels = get_filtered_roi_labels(sys_val, None)
    return (
        gr.Dropdown(choices=subs, value=SUBSYSTEM_ALL),
        gr.CheckboxGroup(choices=roi_labels, value=[]),
    )


def on_subsystem_change(
    system: Optional[str],
    subsystem: Optional[str],
) -> gr.CheckboxGroup:
    """Cascade: Sub-system changed -> update checklist."""
    sys_val = system if system else None
    sub_val = subsystem if subsystem else SUBSYSTEM_ALL
    roi_labels = get_filtered_roi_labels(sys_val, sub_val)
    return gr.CheckboxGroup(choices=roi_labels, value=[])


def on_render(
    checked_rois: Optional[List[str]],
    indices_text: Optional[str],
    legend_mode: Optional[str],
    cmap: Optional[str],
    alpha: Optional[float],
    show_brain: Optional[bool],
) -> Tuple[go.Figure, str]:
    """Main render callback: combine checked ROIs + explicit indices."""
    t0 = time.time()
    info_parts: List[str] = []

    if checked_rois is None:
        checked_rois = []
    if indices_text is None:
        indices_text = ""
    if legend_mode is None:
        legend_mode = DEFAULT_LEGEND_MODE
    if cmap is None:
        cmap = DEFAULT_CMAP
    if alpha is None:
        alpha = DEFAULT_ALPHA
    if show_brain is None:
        show_brain = True

    try:
        selected: set = set()

        for label in checked_rois:
            if label in _LABEL_TO_INDEX:
                selected.add(_LABEL_TO_INDEX[label])
            else:
                logger.warning("Unknown checklist label: %s", label)

        if selected:
            info_parts.append(f"Checked: {len(selected)} ROIs")

        if indices_text.strip():
            tokens = [
                token.strip()
                for token in indices_text.replace(";", ",").split(",")
                if token.strip()
            ]
            explicit: set = set()
            for token in tokens:
                if token.isdigit():
                    idx = int(token)
                    if 0 <= idx < len(_LABELS_DF):
                        explicit.add(idx)
                    else:
                        info_parts.append(f"Out-of-range: {idx}")
                else:
                    info_parts.append(f"Non-numeric skipped: '{token}'")
            if explicit:
                info_parts.append(f"Explicit: +{len(explicit)} ROIs")
            selected |= explicit

        if not selected:
            return go.Figure(), "Nothing selected. Tick ROIs or enter indices."

        final_indices = sorted(selected)
        info_parts.insert(0, f"Total: {len(final_indices)} ROIs")

        fig = build_figure(
            selected_indices=final_indices,
            legend_mode=legend_mode,
            cmap=cmap,
            alpha=alpha,
            show_brain=show_brain,
        )

        elapsed = time.time() - t0
        info_parts.append(f"Rendered in {elapsed:.2f}s")
        return fig, " | ".join(info_parts)

    except (ValueError, RuntimeError) as exc:
        return go.Figure(), f"Error: {exc}"
    except Exception as exc:
        logger.exception("Render failed")
        return go.Figure(), f"Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Build the Gradio Blocks interface — minimalist, viewer-left / controls-right."""
    systems = get_systems()
    n_rois = len(_LABELS_DF) if _LABELS_DF is not None else 0

    theme = (
        gr.themes.Base(
            primary_hue="slate",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        )
        .set(
            body_background_fill=_CLR_BG,
            body_text_color=_CLR_TEXT,
            block_background_fill=_CLR_SURFACE,
            block_border_width="1px",
            block_border_color=_CLR_BORDER,
            block_title_text_color=_CLR_TEXT,
            block_label_text_color=_CLR_TEXT_SECONDARY,
            input_background_fill=_CLR_BG,
            input_border_color=_CLR_BORDER,
            input_border_color_focus=_CLR_BORDER_FOCUS,
            button_primary_background_fill=_CLR_ACCENT,
            button_primary_background_fill_hover=_CLR_ACCENT_HOVER,
            button_primary_text_color="#ffffff",
            button_secondary_background_fill="#ffffff",
            button_secondary_background_fill_hover=_CLR_HOVER_BG,
            button_secondary_border_color=_CLR_BORDER,
            button_secondary_text_color=_CLR_TEXT_SECONDARY,
        )
    )

    with gr.Blocks(
        title="CIMT Volumetric 3D Explorer",
        theme=theme,
        css=APP_CSS,
    ) as app:

        # ── HEADER ──────────────────────────────────────────────────────
        with gr.Row(elem_classes=["app-header"]):
            gr.Markdown(
                "# CIMT Volumetric 3D Explorer\n"
                "448-ROI Atlas · Interactive Mesh Rendering"
            )

        # ── MAIN: Viewer (left) + Sidebar (right) ───────────────────────
        with gr.Row(equal_height=True, elem_classes=["main-layout"]):

            # LEFT — 3D Viewer
            with gr.Column(scale=4, elem_classes=["viewer-panel"]):
                plot_output = gr.Plot(
                    label=None,
                    value=build_initial_figure(),
                    show_label=False,
                    elem_classes=["plot-container"],
                )

            # RIGHT — Control Sidebar
            with gr.Column(scale=1, min_width=320, elem_classes=["sidebar"]):

                # Section 1: Navigation
                with gr.Column(elem_classes=["section-block"]):
                    gr.Markdown(
                        "<div class='section-label'>Navigation</div>"
                    )
                    system_dd = gr.Dropdown(
                        choices=systems,
                        value=None,
                        label="Functional System",
                    )
                    subsystem_dd = gr.Dropdown(
                        choices=[SUBSYSTEM_ALL],
                        value=SUBSYSTEM_ALL,
                        label="Sub-system",
                    )

                # Section 2: Selection
                with gr.Column(elem_classes=["section-block"]):
                    gr.Markdown(
                        "<div class='section-label'>Regions</div>"
                    )
                    roi_checklist = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label=None,
                        show_label=False,
                        elem_classes=["roi-checklist"],
                    )

                # Section 3: Render
                with gr.Column(elem_classes=["section-block"]):
                    render_btn = gr.Button(
                        "Render",
                        variant="primary",
                        size="lg",
                    )
                    status_box = gr.Textbox(
                        label=None,
                        show_label=False,
                        interactive=False,
                        lines=1,
                        placeholder="Awaiting selection…",
                        elem_classes=["status-bar"],
                    )

                # Section 4: Advanced
                with gr.Accordion("Advanced", open=False):
                    indices_input = gr.Textbox(
                        placeholder="e.g. 446, 447",
                        label="Extra Indices",
                        info="Comma-separated",
                    )
                    legend_mode_dd = gr.Dropdown(
                        choices=LEGEND_MODE_CHOICES,
                        value=DEFAULT_LEGEND_MODE,
                        label="Legend Mode",
                    )
                    cmap_dd = gr.Dropdown(
                        choices=CMAP_CHOICES,
                        value=DEFAULT_CMAP,
                        label="Colormap",
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=DEFAULT_ALPHA,
                        label="Opacity",
                    )
                    brain_toggle = gr.Checkbox(
                        value=True,
                        label="Brain Mesh",
                    )

        # ── FOOTER ──────────────────────────────────────────────────────
        gr.Markdown(
            f"<div class='app-footer'>"
            f"{n_rois} ROIs · {len(systems)} Systems · "
            f"Max {MAX_ROIS_RENDER} per render"
            f"</div>"
        )

        # ── EVENT WIRING ────────────────────────────────────────────────

        system_dd.change(
            fn=on_system_change,
            inputs=[system_dd],
            outputs=[subsystem_dd, roi_checklist],
        )
        subsystem_dd.change(
            fn=on_subsystem_change,
            inputs=[system_dd, subsystem_dd],
            outputs=[roi_checklist],
        )
        render_btn.click(
            fn=on_render,
            inputs=[
                roi_checklist,
                indices_input,
                legend_mode_dd,
                cmap_dd,
                alpha_slider,
                brain_toggle,
            ],
            outputs=[plot_output, status_box],
        )

    return app


# ---------------------------------------------------------------------------
# Global Initialization (HF Spaces)
# ---------------------------------------------------------------------------
logger.info("Initializing CIMT Explorer...")
load_atlas_data()

logger.info("Building interface...")
demo = create_app()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"),
        css=APP_CSS,
    )


