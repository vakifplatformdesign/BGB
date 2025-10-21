# BGB

## Textile Analysis Application

This repository includes a Streamlit application (`textile_analysis_app.py`) for comparing textile images under multiple lighting conditions. The app measures color differences, evaluates repeating patterns, and produces a structured PDF report summarizing all metrics and visualizations.

### Features

- Upload reference and test textile images (PNG, JPG, JPEG, TIFF).
- Optional draggable circular Region of Interest (ROI) using `streamlit-drawable-canvas`.
- Color analysis under simulated D65, UV, and TL84 lighting with ΔE76/ΔE2000 metrics.
- Pattern analysis with Fourier spectra, Gabor filters, and edge comparisons.
- Automatic PDF report compilation with all plots and derived metrics.

### Getting Started

1. **Install Dependencies**

   ```bash
   pip install streamlit streamlit-drawable-canvas opencv-python-headless matplotlib numpy pillow scikit-image fpdf2
   ```

2. **Run the Application**

   ```bash
   streamlit run textile_analysis_app.py
   ```

3. **Usage**

   - Upload both the reference and test images.
   - (Optional) Draw a circular ROI on each image; the overlapping area will be analyzed.
   - Click **Run Analysis** to generate metrics, visualizations, and download the PDF report.

The generated plots and report are stored in a temporary directory for the duration of the Streamlit session.
