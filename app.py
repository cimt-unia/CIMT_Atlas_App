# app.py
"""
CIMT Volumetric 3D Explorer — Gradio Application
=================================================
3-level cascade (System -> Hemisphere -> Sub-system) narrows 448 ROIs
to a small checklist. The user manually ticks the ROIs to render.
Extra indices can be added via Advanced.
Layout: 3D plot dominates the top; controls are compact below.
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
BRAIN_COLOR: str = "#d4c5b9"
HEMISPHERE_ALL: str = "All"
SUBSYSTEM_ALL: str = "All Sub-systems"
AUTO_FULL_LEGEND_THRESHOLD: int = 12

CMAP_CHOICES: List[str] = [
    "plasma", "viridis", "inferno", "coolwarm", "tab20", "Set1", "Set2",
]

LEGEND_MODE_CHOICES: List[str] = [
    "auto", "full", "region_full_name", "roi_name",
]

APP_CSS: str = """
    .roi-checklist > div {
        max-height: 220px;
        overflow-y: auto;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px;
    }
    .roi-checklist label {
        font-size: 12px;
    }
    .status-box textarea {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 11px;
    }
    .plot-container {
        min-height: 75vh;
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


def get_hemispheres(system: Optional[str]) -> List[str]:
    """Hemisphere options for a given system."""
    assert _LABELS_DF is not None
    df = _LABELS_DF
    if system:
        df = df[df["functional_system"] == system]
    hemis = sorted(df["hemisphere"].dropna().unique().tolist())
    return [HEMISPHERE_ALL] + hemis


def get_subsystems(
    system: Optional[str], hemisphere: Optional[str]
) -> List[str]:
    """Sub-system options filtered by system + hemisphere."""
    assert _LABELS_DF is not None
    df = _LABELS_DF
    if system:
        df = df[df["functional_system"] == system]
    if hemisphere and hemisphere != HEMISPHERE_ALL:
        df = df[df["hemisphere"] == hemisphere]
    subs = sorted(df["sub_system"].dropna().unique().tolist())
    return [SUBSYSTEM_ALL] + subs


def get_filtered_roi_labels(
    system: Optional[str],
    hemisphere: Optional[str],
    subsystem: Optional[str],
) -> List[str]:
    """Return sorted display labels for ROIs matching the cascade filters."""
    assert _LABELS_DF is not None
    df = _LABELS_DF
    if system:
        df = df[df["functional_system"] == system]
    if hemisphere and hemisphere != HEMISPHERE_ALL:
        df = df[df["hemisphere"] == hemisphere]
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

    # Title.
    title_text = ""

    fig.update_layout(
        title=dict(
            text=f"<b>{title_text}</b>",
            y=0.98,
            x=0.5,
            xanchor="center",
            font=dict(size=18, family="Helvetica, Arial, sans-serif"),
        ),
        width=1200,
        height=850,
        paper_bgcolor="white",
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
            bgcolor="white",
            aspectmode="data",
        ),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e0e0e0",
            borderwidth=1,
            font=dict(size=10),
            title=dict(text="<b>Regions</b>", font=dict(size=11)),
            itemsizing="constant",
        ),
        margin=dict(t=50, b=0, l=0, r=0),
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
            opacity=0.15,
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(
            text="",
            y=0.98,
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Helvetica, Arial, sans-serif"),
        ),
        width=1200,
        height=850,
        paper_bgcolor="white",
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
            bgcolor="white",
            aspectmode="data",
        ),
        margin=dict(t=50, b=0, l=0, r=0),
    )

    return fig


# ---------------------------------------------------------------------------
# Gradio Callbacks
# ---------------------------------------------------------------------------

def on_system_change(
    system: Optional[str],
) -> Tuple[gr.Dropdown, gr.Dropdown, gr.CheckboxGroup]:
    """Cascade: System changed -> update Hemisphere, Sub-system, checklist."""
    sys_val = system if system else None
    hemis = get_hemispheres(sys_val)
    subs = get_subsystems(sys_val, None)
    roi_labels = get_filtered_roi_labels(sys_val, None, None)
    return (
        gr.Dropdown(choices=hemis, value=HEMISPHERE_ALL),
        gr.Dropdown(choices=subs, value=SUBSYSTEM_ALL),
        gr.CheckboxGroup(choices=roi_labels, value=[]),
    )


def on_hemisphere_change(
    system: Optional[str],
    hemisphere: Optional[str],
) -> Tuple[gr.Dropdown, gr.CheckboxGroup]:
    """Cascade: Hemisphere changed -> update Sub-system + checklist."""
    sys_val = system if system else None
    hemi_val = hemisphere if hemisphere else HEMISPHERE_ALL
    subs = get_subsystems(sys_val, hemi_val)
    roi_labels = get_filtered_roi_labels(sys_val, hemi_val, None)
    return (
        gr.Dropdown(choices=subs, value=SUBSYSTEM_ALL),
        gr.CheckboxGroup(choices=roi_labels, value=[]),
    )


def on_subsystem_change(
    system: Optional[str],
    hemisphere: Optional[str],
    subsystem: Optional[str],
) -> gr.CheckboxGroup:
    """Cascade: Sub-system changed -> update checklist."""
    sys_val = system if system else None
    hemi_val = hemisphere if hemisphere else HEMISPHERE_ALL
    sub_val = subsystem if subsystem else SUBSYSTEM_ALL
    roi_labels = get_filtered_roi_labels(sys_val, hemi_val, sub_val)
    return gr.CheckboxGroup(choices=roi_labels, value=[])


def on_select_all(
    system: Optional[str],
    hemisphere: Optional[str],
    subsystem: Optional[str],
) -> gr.CheckboxGroup:
    """Select all currently filtered ROIs."""
    sys_val = system if system else None
    hemi_val = hemisphere if hemisphere else HEMISPHERE_ALL
    sub_val = subsystem if subsystem else SUBSYSTEM_ALL
    roi_labels = get_filtered_roi_labels(sys_val, hemi_val, sub_val)
    return gr.CheckboxGroup(value=roi_labels)


def on_clear_all() -> gr.CheckboxGroup:
    """Clear all checked ROIs."""
    return gr.CheckboxGroup(value=[])


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
    """Build the Gradio Blocks interface with plot-dominant layout."""
    systems = get_systems()
    n_rois = len(_LABELS_DF) if _LABELS_DF is not None else 0

    with gr.Blocks(title="CIMT Volumetric 3D Explorer") as app:

        # --- TOP: Full-width 3D Plot (dominant) ---
        plot_output = gr.Plot(
            label="3D Volumetric View",
            value=build_initial_figure(),
            elem_classes=["plot-container"],
        )

        # --- BOTTOM: Compact controls ---
        with gr.Row():
            system_dd = gr.Dropdown(
                choices=systems,
                value=None,
                label="Functional System",
                scale=2,
            )
            hemisphere_dd = gr.Dropdown(
                choices=[HEMISPHERE_ALL, "Left", "Right"],
                value=HEMISPHERE_ALL,
                label="Hemisphere",
                scale=1,
            )
            subsystem_dd = gr.Dropdown(
                choices=[SUBSYSTEM_ALL],
                value=SUBSYSTEM_ALL,
                label="Sub-system",
                scale=2,
            )
            render_btn = gr.Button(
                "Render", variant="primary", scale=1,
            )

        # ROI Checklist with Select All / Clear buttons
        with gr.Row():
            select_all_btn = gr.Button("Select All", size="sm", scale=1)
            clear_all_btn = gr.Button("Clear All", size="sm", scale=1)

        roi_checklist = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="ROIs (tick to select)",
            elem_classes=["roi-checklist"],
        )

        # Status bar (full width)
        status_box = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2,
            elem_classes=["status-box"],
        )

        # Advanced options (full width, collapsed)
        with gr.Accordion("Advanced", open=False):
            indices_input = gr.Textbox(
                placeholder="e.g. 446, 447",
                label="Extra Indices (comma-separated)",
                info="Appended to checked ROIs. Works standalone too.",
            )
            with gr.Row():
                legend_mode_dd = gr.Dropdown(
                    choices=LEGEND_MODE_CHOICES,
                    value=DEFAULT_LEGEND_MODE,
                    label="Legend Labels",
                    info="auto: full names for small selections, ROI codes for large",
                )
                cmap_dd = gr.Dropdown(
                    choices=CMAP_CHOICES,
                    value=DEFAULT_CMAP,
                    label="Colormap",
                )
            with gr.Row():
                alpha_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=DEFAULT_ALPHA,
                    label="Surface Opacity",
                )
                brain_toggle = gr.Checkbox(
                    value=True,
                    label="Brain Reference Mesh",
                )

        gr.Markdown(
            f"*{n_rois} ROIs | {len(systems)} systems | "
            f"Max {MAX_ROIS_RENDER} per render*"
        )

        # --- Event Wiring ---

        system_dd.change(
            fn=on_system_change,
            inputs=[system_dd],
            outputs=[hemisphere_dd, subsystem_dd, roi_checklist],
        )
        hemisphere_dd.change(
            fn=on_hemisphere_change,
            inputs=[system_dd, hemisphere_dd],
            outputs=[subsystem_dd, roi_checklist],
        )
        subsystem_dd.change(
            fn=on_subsystem_change,
            inputs=[system_dd, hemisphere_dd, subsystem_dd],
            outputs=[roi_checklist],
        )
        select_all_btn.click(
            fn=on_select_all,
            inputs=[system_dd, hemisphere_dd, subsystem_dd],
            outputs=[roi_checklist],
        )
        clear_all_btn.click(
            fn=on_clear_all,
            inputs=[],
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
