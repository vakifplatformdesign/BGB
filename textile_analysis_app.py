import io
import os
import math
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from PIL import Image
from skimage import color, filters

try:
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas
except ImportError:  # pragma: no cover - streamlit is optional for non-GUI execution
    st = None
    st_canvas = None


LIGHTING_MATRICES: Dict[str, np.ndarray] = {
    "D65": np.array(
        [
            [1.05, 0.02, -0.03],
            [0.01, 0.99, 0.02],
            [-0.02, 0.03, 1.04],
        ]
    ),
    "UV": np.array(
        [
            [0.90, 0.05, 0.05],
            [0.04, 0.85, 0.11],
            [0.05, 0.08, 0.87],
        ]
    ),
    "TL84": np.array(
        [
            [1.10, -0.04, -0.02],
            [-0.03, 1.06, -0.01],
            [0.02, -0.02, 0.98],
        ]
    ),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_image(uploaded_file: io.BytesIO) -> np.ndarray:
    """Load an uploaded image into an RGB numpy array."""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def apply_roi_mask(image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return image
    masked = image.copy()
    if image.ndim == 3:
        for c in range(3):
            channel = masked[..., c]
            channel[~mask] = 0
            masked[..., c] = channel
    else:
        masked[~mask] = 0
    return masked


def simulate_lighting(image: np.ndarray, condition: str) -> np.ndarray:
    """Apply a simple spectral transformation matrix to simulate lighting."""
    matrix = LIGHTING_MATRICES.get(condition)
    if matrix is None:
        raise ValueError(f"Unknown lighting condition: {condition}")

    reshaped = image.reshape(-1, 3).astype(np.float32) / 255.0
    transformed = reshaped @ matrix.T
    transformed = np.clip(transformed, 0, 1)
    return (transformed.reshape(image.shape) * 255).astype(np.uint8)


def calculate_delta_e(
    ref_lab: np.ndarray,
    test_lab: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute Delta E metrics and provide maps."""
    if mask is not None:
        mask = mask.astype(bool)
        ref_lab = ref_lab[mask]
        test_lab = test_lab[mask]

    delta_e76_map = color.deltaE_cie76(ref_lab, test_lab)
    try:
        delta_e2000_map = color.deltaE_ciede2000(ref_lab, test_lab)
    except Exception:
        delta_e2000_map = np.zeros_like(delta_e76_map)

    delta_e76 = float(np.mean(delta_e76_map))
    delta_e2000 = float(np.mean(delta_e2000_map))

    return delta_e76, delta_e2000, delta_e76_map, delta_e2000_map


def plot_histogram_comparison(
    reference: np.ndarray,
    test: np.ndarray,
    color_space: str,
    output_path: str,
    mask: Optional[np.ndarray] = None,
) -> None:
    if mask is not None:
        ref_values = reference[mask] if reference.ndim == 3 else reference[mask]
        test_values = test[mask] if test.ndim == 3 else test[mask]
    else:
        ref_values = (
            reference.reshape(-1, reference.shape[-1])
            if reference.ndim == 3
            else reference.reshape(-1)
        )
        test_values = (
            test.reshape(-1, test.shape[-1]) if test.ndim == 3 else test.reshape(-1)
        )

    plt.figure(figsize=(10, 4))
    if reference.ndim == 3:
        labels = ["R", "G", "B"] if color_space.upper() == "RGB" else ["L*", "a*", "b*"]
        for idx, label in enumerate(labels):
            plt.hist(
                ref_values[:, idx],
                bins=256,
                alpha=0.4,
                label=f"Reference {label}",
                color=["r", "g", "b"][idx % 3],
            )
            plt.hist(
                test_values[:, idx],
                bins=256,
                alpha=0.4,
                label=f"Test {label}",
                histtype="step",
                color=["r", "g", "b"][idx % 3],
            )
    else:
        plt.hist(ref_values, bins=256, alpha=0.5, label="Reference", color="gray")
        plt.hist(test_values, bins=256, alpha=0.5, label="Test", color="black")

    plt.title(f"{color_space} Histogram Comparison")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_side_by_side(images: List[np.ndarray], titles: List[str], output_path: str) -> None:
    cols = len(images)
    plt.figure(figsize=(5 * cols, 5))
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, cols, idx + 1)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fourier_spectrum(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude_spectrum.astype(np.uint8)


def gabor_responses(image: np.ndarray, frequencies: List[float], thetas: List[float]) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
    responses = {}
    for frequency in frequencies:
        for theta in thetas:
            real, imag = filters.gabor(gray, frequency=frequency, theta=theta)
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            key = f"freq_{frequency:.2f}_theta_{math.degrees(theta):.0f}"
            responses[key] = magnitude.astype(np.uint8)
    return responses


def edge_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


def analyze_colors(
    reference: np.ndarray,
    test: np.ndarray,
    results_dir: str,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict:
    color_dir = ensure_dir(os.path.join(results_dir, "color"))
    color_results: Dict[str, Dict] = {
        "delta_e76": {},
        "delta_e2000": {},
        "delta_maps": {},
        "lighting_visuals": {},
    }

    for label in LIGHTING_MATRICES.keys():
        ref_light = simulate_lighting(reference, label)
        test_light = simulate_lighting(test, label)

        ref_lab = cv2.cvtColor(ref_light, cv2.COLOR_RGB2LAB).astype(np.float32)
        test_lab = cv2.cvtColor(test_light, cv2.COLOR_RGB2LAB).astype(np.float32)

        if roi_mask is not None:
            delta_e76, delta_e2000, delta_map, delta2000_map = calculate_delta_e(
                ref_lab, test_lab, roi_mask
            )
        else:
            delta_e76, delta_e2000, delta_map, delta2000_map = calculate_delta_e(
                ref_lab.reshape(-1, 3), test_lab.reshape(-1, 3)
            )

        color_results["delta_e76"][label] = delta_e76
        color_results["delta_e2000"][label] = delta_e2000

        if roi_mask is not None:
            full_map = np.zeros(reference.shape[:2], dtype=np.float32)
            full_map[roi_mask] = delta_map
            delta_map_image = full_map
        else:
            delta_map_image = delta_map.reshape(reference.shape[:2])

        delta_path = os.path.join(color_dir, f"delta_e76_{label}.png")
        plt.figure(figsize=(5, 5))
        plt.imshow(delta_map_image, cmap="inferno")
        plt.colorbar(label="ΔE76")
        plt.title(f"ΔE76 Map - {label}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(delta_path)
        plt.close()
        color_results["delta_maps"][label] = delta_path

        visual_path = os.path.join(color_dir, f"visual_{label}.png")
        plot_side_by_side(
            [ref_light, test_light, cv2.absdiff(ref_light, test_light)],
            [f"Reference - {label}", f"Test - {label}", "Abs Difference"],
            visual_path,
        )
        color_results["lighting_visuals"][label] = visual_path

    rgb_hist_path = os.path.join(color_dir, "rgb_histogram.png")
    lab_hist_path = os.path.join(color_dir, "lab_histogram.png")

    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
    test_lab = cv2.cvtColor(test, cv2.COLOR_RGB2LAB)
    plot_histogram_comparison(reference, test, "RGB", rgb_hist_path, roi_mask)
    plot_histogram_comparison(reference_lab, test_lab, "LAB", lab_hist_path, roi_mask)

    color_results["rgb_hist_path"] = rgb_hist_path
    color_results["lab_hist_path"] = lab_hist_path
    return color_results


def analyze_patterns(
    reference: np.ndarray,
    test: np.ndarray,
    results_dir: str,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict:
    pattern_dir = ensure_dir(os.path.join(results_dir, "pattern"))

    ref_roi = apply_roi_mask(reference, roi_mask)
    test_roi = apply_roi_mask(test, roi_mask)

    fourier_ref = fourier_spectrum(ref_roi)
    fourier_test = fourier_spectrum(test_roi)
    fourier_path = os.path.join(pattern_dir, "fourier.png")
    plot_side_by_side([fourier_ref, fourier_test], ["Reference", "Test"], fourier_path)

    frequencies = [0.1, 0.2, 0.3]
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    gabor_ref = gabor_responses(ref_roi, frequencies, thetas)
    gabor_test = gabor_responses(test_roi, frequencies, thetas)

    gabor_paths = {}
    for key in gabor_ref:
        output_path = os.path.join(pattern_dir, f"gabor_{key}.png")
        plot_side_by_side([gabor_ref[key], gabor_test[key]], ["Reference", "Test"], output_path)
        gabor_paths[key] = output_path

    edges_ref = edge_map(ref_roi)
    edges_test = edge_map(test_roi)
    edges_path = os.path.join(pattern_dir, "edges.png")
    plot_side_by_side([edges_ref, edges_test], ["Reference", "Test"], edges_path)

    mask_area = int(np.count_nonzero(roi_mask)) if roi_mask is not None else int(reference.shape[0] * reference.shape[1])

    # Structural comparison metrics
    edge_similarity = float(
        np.sum(edges_ref == edges_test) / (edges_ref.size if edges_ref.size else 1)
    )
    ref_energy = float(np.mean(fourier_ref))
    test_energy = float(np.mean(fourier_test))
    gabor_diff = float(
        np.mean([np.mean(np.abs(gabor_ref[k] - gabor_test[k])) for k in gabor_ref])
    )

    return {
        "fourier_path": fourier_path,
        "gabor_paths": gabor_paths,
        "edges_path": edges_path,
        "metrics": {
            "edge_similarity": edge_similarity,
            "fourier_energy_reference": ref_energy,
            "fourier_energy_test": test_energy,
            "gabor_mean_difference": gabor_diff,
            "roi_pixels": mask_area,
        },
    }


def generate_pdf_report(
    color_results: Dict,
    pattern_results: Dict,
    output_path: str,
) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_section_header(title: str):
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(4)

    def add_subtitle(text: str):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, text, ln=True)
        pdf.ln(2)

    def add_paragraph(text: str):
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, text)
        pdf.ln(2)

    pdf.add_page()
    add_section_header("Textile Analysis Report")
    add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Section 1: Color Analysis
    add_section_header("Section 1: Color Analysis")
    add_subtitle("ΔE Metrics")
    for lighting, value in color_results["delta_e76"].items():
        delta2000 = color_results["delta_e2000"].get(lighting, 0.0)
        add_paragraph(
            f"{lighting}: ΔE76 = {value:.3f}, ΔE2000 = {delta2000:.3f}"
        )

    add_subtitle("RGB Histogram")
    pdf.image(color_results["rgb_hist_path"], w=180)
    add_subtitle("LAB Histogram")
    pdf.image(color_results["lab_hist_path"], w=180)

    add_subtitle("Lighting Visual Comparisons")
    for lighting, path in color_results["lighting_visuals"].items():
        add_paragraph(lighting)
        pdf.image(path, w=180)

    add_subtitle("ΔE76 Maps")
    for lighting, path in color_results["delta_maps"].items():
        add_paragraph(lighting)
        pdf.image(path, w=180)

    # Section 2: Pattern Analysis
    add_section_header("Section 2: Pattern Analysis")
    add_subtitle("Fourier Spectrum")
    pdf.image(pattern_results["fourier_path"], w=180)

    add_subtitle("Gabor Filter Responses")
    for key, path in pattern_results["gabor_paths"].items():
        add_paragraph(key)
        pdf.image(path, w=180)

    add_subtitle("Edge Maps")
    pdf.image(pattern_results["edges_path"], w=180)

    add_subtitle("Pattern Metrics")
    metrics_lines = [
        f"Edge Similarity: {pattern_results['metrics']['edge_similarity']:.4f}",
        f"Fourier Energy (Reference): {pattern_results['metrics']['fourier_energy_reference']:.2f}",
        f"Fourier Energy (Test): {pattern_results['metrics']['fourier_energy_test']:.2f}",
        f"Gabor Mean Difference: {pattern_results['metrics']['gabor_mean_difference']:.2f}",
        f"ROI Pixels Analyzed: {pattern_results['metrics']['roi_pixels']}",
    ]
    add_paragraph("\n".join(metrics_lines))

    pdf.output(output_path)


def resize_to_match(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    if reference.shape[:2] == test.shape[:2]:
        return test
    return cv2.resize(test, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def get_roi_mask_from_canvas(canvas_result, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if canvas_result is None or canvas_result.json_data is None:
        return None
    objects = canvas_result.json_data.get("objects", [])
    for obj in objects[::-1]:
        if obj.get("type") == "circle":
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            radius = obj.get("radius")
            if radius is None:
                width = obj.get("width", 0)
                radius = width / 2
            center_x = int(left + radius)
            center_y = int(top + radius)
            radius = int(radius)
            yy, xx = np.ogrid[: image_shape[0], : image_shape[1]]
            mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
            return mask
    return None


def run_streamlit_app():  # pragma: no cover - UI function
    st.set_page_config(page_title="Textile Color & Pattern Analysis", layout="wide")
    st.title("Textile Color & Pattern Analysis")
    st.write(
        "Upload a reference textile image and a test textile image to compare color and pattern differences under multiple lighting conditions."
    )

    reference_file = st.file_uploader("Reference Image", type=["png", "jpg", "jpeg", "tiff"])
    test_file = st.file_uploader("Test Image", type=["png", "jpg", "jpeg", "tiff"])

    if reference_file and test_file:
        reference_image = load_image(reference_file)
        test_image = load_image(test_file)
        test_image = resize_to_match(reference_image, test_image)

        st.subheader("Region of Interest (ROI)")
        st.write(
            "Optionally draw a circular ROI on both images. If omitted, the full image will be analyzed."
        )

        col1, col2 = st.columns(2)
        roi_mask_ref = roi_mask_test = None

        if st_canvas is not None:
            with col1:
                st.caption("Reference ROI (draw a circle)")
                canvas_ref = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_image=Image.fromarray(reference_image),
                    height=reference_image.shape[0],
                    width=reference_image.shape[1],
                    drawing_mode="circle",
                    key="canvas_ref",
                )
                roi_mask_ref = get_roi_mask_from_canvas(canvas_ref, reference_image.shape[:2])
            with col2:
                st.caption("Test ROI (draw a circle)")
                canvas_test = st_canvas(
                    fill_color="rgba(0, 0, 255, 0.3)",
                    stroke_width=2,
                    stroke_color="#0000FF",
                    background_image=Image.fromarray(test_image),
                    height=test_image.shape[0],
                    width=test_image.shape[1],
                    drawing_mode="circle",
                    key="canvas_test",
                )
                roi_mask_test = get_roi_mask_from_canvas(canvas_test, test_image.shape[:2])
        else:
            st.warning(
                "streamlit-drawable-canvas is not installed. Install it to enable draggable ROIs."
            )

        if roi_mask_ref is not None and roi_mask_test is not None:
            roi_mask = roi_mask_ref & roi_mask_test
        elif roi_mask_ref is not None:
            roi_mask = roi_mask_ref
        elif roi_mask_test is not None:
            roi_mask = roi_mask_test
        else:
            roi_mask = None

        if st.button("Run Analysis"):
            with st.spinner("Analyzing textile images..."):
                results_dir = tempfile.mkdtemp(prefix="textile_analysis_")
                color_results = analyze_colors(reference_image, test_image, results_dir, roi_mask)
                pattern_results = analyze_patterns(reference_image, test_image, results_dir, roi_mask)
                pdf_path = os.path.join(results_dir, "Textile_Analysis_Report.pdf")
                generate_pdf_report(color_results, pattern_results, pdf_path)

            st.success("Analysis complete!")

            st.subheader("Color Analysis Results")
            for lighting in LIGHTING_MATRICES.keys():
                delta76 = color_results["delta_e76"][lighting]
                delta2000 = color_results["delta_e2000"][lighting]
                st.write(f"**{lighting}** - ΔE76: {delta76:.2f}, ΔE2000: {delta2000:.2f}")
                st.image(color_results["lighting_visuals"][lighting], use_column_width=True)
                st.image(color_results["delta_maps"][lighting], caption=f"ΔE76 Map - {lighting}")

            st.image(color_results["rgb_hist_path"], caption="RGB Histogram", use_column_width=True)
            st.image(color_results["lab_hist_path"], caption="LAB Histogram", use_column_width=True)

            st.subheader("Pattern Analysis Results")
            st.image(pattern_results["fourier_path"], caption="Fourier Spectrum", use_column_width=True)
            for key, path in pattern_results["gabor_paths"].items():
                st.image(path, caption=f"Gabor Response {key}", use_column_width=True)
            st.image(pattern_results["edges_path"], caption="Edge Maps", use_column_width=True)

            st.json(pattern_results["metrics"])

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_file,
                    file_name="Textile_Analysis_Report.pdf",
                    mime="application/pdf",
                )

    else:
        st.info("Please upload both images to begin the analysis.")


if __name__ == "__main__":
    if st is not None:
        run_streamlit_app()
    else:
        print(
            "Streamlit is not installed. Install streamlit and streamlit-drawable-canvas to run the GUI."
        )
