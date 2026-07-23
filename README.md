# CIMT Volumetric 3D Explorer

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-blue?logo=render)](https://cimt-atlas-app.onrender.com/)


Interactive 3D visualization of the CIMT 448-ROI Atlas. This application enables researchers to explore functional brain systems through a hierarchical cascade filter (System, Hemisphere, Sub-system), select regions of interest, and render them as interactive 3D meshes in the browser.

**Live Application:** [https://cimt-atlas-app.onrender.com/](https://cimt-atlas-app.onrender.com/)

## Features

-   **Hierarchical Filtering:** Narrow 448 ROIs using a three-level cascade: Functional System, Hemisphere, and Sub-system.
-   **Volumetric Rendering:** Plotly-based 3D visualization with configurable opacity, colormaps, and legend display modes.
-   **ROI Selection Management:** Checkbox groups with bulk selection controls and support for manual index entry via the Advanced panel.
-   **Anatomical Context:** Optional fsaverage5 pial surface overlay for spatial reference.
-   **Cloud Deployment:** Configured for Render Web Services, Hugging Face Spaces, or Docker environments.

¡

## Deployment on Render

This application is configured to run as a Web Service on Render. It requires a persistent runtime for mesh extraction and cannot be deployed as a Static Site.

### Prerequisites

-   Active Render account.
-   Repository connected to Render via GitHub integration.

### Service Configuration

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| Service Type | Web Service | Required for dynamic Python execution |
| Runtime | Python 3 | Auto-detected from repository |
| Build Command | `pip install -r requirements.txt` | Installs pinned dependencies |
| Start Command | `python app.py` | Launches Gradio ASGI server |
| Instance Type | Standard (2 GB RAM) | Minimum recommended for mesh processing |
| Environment Variable | `PYTHON_VERSION=3.11.0` | Pins interpreter version |
