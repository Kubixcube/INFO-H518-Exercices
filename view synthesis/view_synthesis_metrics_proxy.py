from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ===================== CONFIGURATION ========================

LEFT_IMAGE_PATH = "cone0.png"

SYNTH_MID_IMAGE_PATH = "out_view_synthesis/view_fused.png"

DISPARITY_LEFT_PATH = "cones_disparity_occlusion_filled_smoothed.png"

DISP_IS_8BIT = True

DISP_MIN = -51
DISP_MAX = -17

DISP_SIGN_LEFT = -1.0

# --- Mid-view position used for synthesis ---
ALPHA = 0.5 

# --- Validity mask ---
BLACK_THRESH = 5  # pixels <= threshold are considered invalid (holes/out-of-image)

# --- IV-PSNR-like shift tolerance ---
SHIFT_RADIUS = 2  # (2R+1)x(2R+1) window

# --- Optional global color offset compensation (per-channel bias) ---
USE_GLOBAL_COLOR_OFFSET = True

# --- Output folder for debug images + plots ---
SAVE_DEBUG = True
DEBUG_DIR = "out_metrics_proxy"


# ============================================================

@dataclass
class Metric:
    mse: float
    psnr_db: float


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_disp_raw(path: str) -> np.ndarray:
    disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp is None:
        raise FileNotFoundError(f"Could not read disparity: {path}")
    if disp.ndim == 3:
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
    return disp.astype(np.float32)


def bgr_to_luma_y(bgr: np.ndarray) -> np.ndarray:
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r


def rescale_disp_from_8bit(disp8: np.ndarray, disp_min: float, disp_max: float) -> np.ndarray:
    return (disp8 / 255.0) * (disp_max - disp_min) + disp_min


def decode_disparity(disp_raw: np.ndarray) -> np.ndarray:
    """
    Decode disparity to pixel units, then apply the sign convention.
    """
    if DISP_IS_8BIT:
        disp = rescale_disp_from_8bit(disp_raw, DISP_MIN, DISP_MAX)
    else:
        disp = disp_raw.copy()

    disp = DISP_SIGN_LEFT * disp
    return disp


def ensure_same_size(ref: np.ndarray, img: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
    if img.shape[:2] == ref.shape[:2]:
        return img
    return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=interp)


# ============================================================

def reproject_mid_to_left(
    mid_bgr: np.ndarray,
    disp_left: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproject synthesized mid-view back to LEFT coordinates using LEFT disparity.

    Geometry (after rectification) with disparity d = x_L - x_R:
      x_mid = x_L - alpha * d_L(x_L)

    Therefore, to reconstruct LEFT pixels from the MID image:
      MID sampling coordinate for each left pixel is:
        x_src_mid = x_L - alpha * d_L(x_L)

    Returns:
        reproj_bgr (BGR) in LEFT coordinates
        inbounds_mask: True where sampling coords are inside MID image bounds
    """
    h, w = disp_left.shape

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_src = xs - float(alpha) * disp_left
    y_src = ys

    inbounds = (x_src >= 0.0) & (x_src <= (w - 1.0)) & np.isfinite(disp_left)

    reproj = cv2.remap(
        mid_bgr,
        x_src.astype(np.float32),
        y_src.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    reproj[~inbounds] = 0
    return reproj, inbounds


# ============================================================

def build_valid_mask(ref_bgr: np.ndarray, img_bgr: np.ndarray, black_thresh: int) -> np.ndarray:
    """
    Evaluate only pixels that are not "almost black" in BOTH images.
    This ignores holes (black) and out-of-image regions.
    """
    ref_y = bgr_to_luma_y(ref_bgr)
    img_y = bgr_to_luma_y(img_bgr)
    valid = (ref_y > float(black_thresh)) & (img_y > float(black_thresh))
    return valid


def global_color_offset_comp(ref_bgr: np.ndarray, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Remove constant per-channel bias on valid pixels:
      img' = img + mean(ref - img) over valid pixels (per channel)
    """
    img = img_bgr.astype(np.float32)
    ref = ref_bgr.astype(np.float32)
    valid = mask.astype(bool)

    if valid.sum() == 0:
        return img

    out = img.copy()
    for c in range(3):
        diff = ref[..., c] - img[..., c]
        bias = float(diff[valid].mean())
        out[..., c] = np.clip(out[..., c] + bias, 0.0, 255.0)

    return out


def mse_pixelwise(ref: np.ndarray, img: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return float("nan")

    diff = (ref - img).astype(np.float32)
    if diff.ndim == 3:
        diff2 = (diff * diff).sum(axis=2)
    else:
        diff2 = diff * diff

    return float(diff2[valid].mean())


def mse_shift_tolerant(ref: np.ndarray, img: np.ndarray, mask: np.ndarray, radius: int) -> float:
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return float("nan")

    ref_f = ref.astype(np.float32)
    img_f = img.astype(np.float32)

    channel_sum = (ref_f.ndim == 3)
    pad = radius

    if channel_sum:
        img_pad = np.pad(img_f, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    else:
        img_pad = np.pad(img_f, ((pad, pad), (pad, pad)), mode="edge")

    h, w = valid.shape
    best = np.full((h, w), np.inf, dtype=np.float32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = pad + dy
            x0 = pad + dx

            if channel_sum:
                patch = img_pad[y0:y0 + h, x0:x0 + w, :]
                d = ref_f - patch
                se = (d * d).sum(axis=2)
            else:
                patch = img_pad[y0:y0 + h, x0:x0 + w]
                d = ref_f - patch
                se = d * d

            best = np.minimum(best, se)

    return float(best[valid].mean())


def mse_to_psnr(mse: float, peak: float = 255.0) -> float:
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((peak * peak) / mse)


def compute_metrics(ref_bgr: np.ndarray, img_bgr: np.ndarray, mask: np.ndarray, radius: int) -> Dict[str, Metric]:
    out: Dict[str, Metric] = {}

    mse_rgb = mse_pixelwise(ref_bgr.astype(np.float32), img_bgr.astype(np.float32), mask)
    out["PSNR (RGB)"] = Metric(mse=mse_rgb, psnr_db=mse_to_psnr(mse_rgb))

    iv_mse_rgb = mse_shift_tolerant(ref_bgr, img_bgr, mask, radius)
    out["IV-PSNR (RGB, shift-tolerant)"] = Metric(mse=iv_mse_rgb, psnr_db=mse_to_psnr(iv_mse_rgb))

    ref_y = bgr_to_luma_y(ref_bgr)
    img_y = bgr_to_luma_y(img_bgr)

    mse_y = mse_pixelwise(ref_y, img_y, mask)
    out["PSNR (Luma Y)"] = Metric(mse=mse_y, psnr_db=mse_to_psnr(mse_y))

    iv_mse_y = mse_shift_tolerant(ref_y, img_y, mask, radius)
    out["IV-PSNR (Luma Y, shift-tolerant)"] = Metric(mse=iv_mse_y, psnr_db=mse_to_psnr(iv_mse_y))

    return out


# ============================================================

def plot_psnr_bars(metrics: Dict[str, Metric], out_path: Path) -> None:
    labels = list(metrics.keys())
    values = [metrics[k].psnr_db for k in labels]

    plt.figure(figsize=(10, 4))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("PSNR (dB)")
    plt.title("View Synthesis Quality (Proxy): PSNR vs IV-PSNR-like")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_histogram(ref_bgr: np.ndarray, img_bgr: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return

    ref = ref_bgr.astype(np.float32)
    img = img_bgr.astype(np.float32)
    d = ref - img
    se = (d * d).sum(axis=2)
    data = se[valid].ravel()

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=80)
    plt.xlabel("Per-pixel squared error (sum over RGB)")
    plt.ylabel("Count")
    plt.title("Error distribution on valid pixels")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ===================== MAIN ==================================

def main() -> None:
    print("=== INFO-H518 — Q_V3 — View Synthesis Metrics (PROXY, adapted) ===")

    # Load images
    left_ref_bgr = read_bgr(LEFT_IMAGE_PATH)
    synth_mid_bgr = read_bgr(SYNTH_MID_IMAGE_PATH)

    # Ensure consistent sizes
    synth_mid_bgr = ensure_same_size(left_ref_bgr, synth_mid_bgr, interp=cv2.INTER_LINEAR)

    # Load & decode disparity
    disp_raw = read_disp_raw(DISPARITY_LEFT_PATH)
    disp_left = decode_disparity(disp_raw)

    if disp_left.shape[:2] != left_ref_bgr.shape[:2]:
        disp_left = cv2.resize(
            disp_left,
            (left_ref_bgr.shape[1], left_ref_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    print(f"Decoded left disparity range (after rescale+sign): {disp_left.min():.3f} .. {disp_left.max():.3f}")
    print(f"DISP_IS_8BIT={DISP_IS_8BIT}, DISP_MIN={DISP_MIN}, DISP_MAX={DISP_MAX}, DISP_SIGN_LEFT={DISP_SIGN_LEFT}")
    print(f"ALPHA={ALPHA}")

    # Reproject mid -> left
    reproj_bgr, inbounds = reproject_mid_to_left(synth_mid_bgr, disp_left, ALPHA)

    # Build validity mask (holes/out-of-bounds)
    valid_mask = build_valid_mask(left_ref_bgr, reproj_bgr, BLACK_THRESH) & inbounds

    # Optional global color offset
    reproj_for_metrics = reproj_bgr
    if USE_GLOBAL_COLOR_OFFSET:
        reproj_for_metrics = global_color_offset_comp(left_ref_bgr, reproj_bgr, valid_mask).astype(np.uint8)

    # Compute metrics
    metrics = compute_metrics(left_ref_bgr, reproj_for_metrics, valid_mask, SHIFT_RADIUS)

    valid_ratio = float(valid_mask.mean())

    print()
    print(f"Reference (real left):       {LEFT_IMAGE_PATH}")
    print(f"Synthesized mid (input):     {SYNTH_MID_IMAGE_PATH}")
    print(f"Disparity (left):            {DISPARITY_LEFT_PATH}")
    print(f"Valid pixel ratio:           {valid_ratio:.3f}")
    print(f"IV-PSNR shift radius R:      {SHIFT_RADIUS}  (window {(2*SHIFT_RADIUS+1)}x{(2*SHIFT_RADIUS+1)})")
    print(f"Global color offset comp.:   {USE_GLOBAL_COLOR_OFFSET}")
    print(f"Black threshold (mask):      {BLACK_THRESH}")
    print()

    for name, m in metrics.items():
        psnr_str = "inf" if math.isinf(m.psnr_db) else f"{m.psnr_db:.3f} dB"
        print(f"{name:35s}  MSE={m.mse:.4f}   PSNR={psnr_str}")

    # Debug outputs
    if SAVE_DEBUG:
        out_dir = Path(DEBUG_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / "reference_left.png"), left_ref_bgr)
        cv2.imwrite(str(out_dir / "reprojected_mid_to_left.png"), reproj_for_metrics)
        cv2.imwrite(str(out_dir / "valid_mask.png"), (valid_mask.astype(np.uint8) * 255))

        plot_psnr_bars(metrics, out_dir / "psnr_barplot.png")
        plot_error_histogram(left_ref_bgr, reproj_for_metrics, valid_mask, out_dir / "error_histogram.png")

        print()
        print(f"Saved debug outputs to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
