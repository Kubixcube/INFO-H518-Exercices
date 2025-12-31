import cv2
import numpy as np
from pathlib import Path

# ===================== CONFIGURATION ========================

LEFT_IMAGE_PATH = "cone0.png"
RIGHT_IMAGE_PATH = "cone1.png"

DISP_LEFT_PATH = "cones_disparity_occlusion_filled_smoothed.png"
DISP_RIGHT_PATH = "disparity_occlusion_filled_smoothedRL.png"

OUTPUT_DIR = "out_view_synthesis"

ALPHA = 0.5

DISP_IS_8BIT = True
DISP_MIN = -51
DISP_MAX = -17

DISP_SIGN_LEFT = -1.0
DISP_SIGN_RIGHT = -1.0

FORWARD_USE_ZBUFFER = True
FORWARD_HOLE_FILL = True

BACKWARD_INTERP = cv2.INTER_LINEAR
BORDER_MODE = cv2.BORDER_CONSTANT
BORDER_VALUE = 0

# ============================================================


def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image non trouvée: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f"Impossible d'écrire: {path}")


def load_disparity(path: str) -> np.ndarray:
    disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp is None:
        raise FileNotFoundError(f"Disparity non trouvée: {path}")

    if disp.ndim == 3:
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)

    return disp.astype(np.float32)


def rescale_disparity_if_needed(disp_raw: np.ndarray, disp_is_8bit: bool, dmin: float, dmax: float) -> np.ndarray:
    if disp_is_8bit:
        return (disp_raw / 255.0) * (dmax - dmin) + dmin
    return disp_raw


def apply_disp_sign(disp: np.ndarray, sign: float) -> np.ndarray:
    return sign * disp


# Forward warp (vectorisé)

def forward_warp_from_left(left_rgb: np.ndarray, disp_px: np.ndarray, alpha: float, use_zbuffer: bool) -> tuple[np.ndarray, np.ndarray]:
    h, w = disp_px.shape
    out = np.zeros_like(left_rgb, dtype=np.uint8)
    valid = np.zeros((h, w), dtype=bool)

    ys, xs = np.indices((h, w))
    xt = np.rint(xs.astype(np.float32) - alpha * disp_px).astype(np.int32)

    in_bounds = (xt >= 0) & (xt < w)
    if not np.any(in_bounds):
        return out, valid

    yv = ys[in_bounds]
    xv = xs[in_bounds]
    xtv = xt[in_bounds]
    dv = disp_px[in_bounds]

    if use_zbuffer:
        depth = np.abs(dv)

        zbuf = np.full((h, w), -np.inf, dtype=np.float32)
        z_at_target = zbuf[yv, xtv]
        keep = depth > z_at_target

        key = yv.astype(np.int64) * w + xtv.astype(np.int64)
        order = np.argsort(key)
        key_s = key[order]
        depth_s = depth[order]
        y_s = yv[order]
        x_s = xv[order]
        xt_s = xtv[order]

        keep_s = np.zeros_like(depth_s, dtype=bool)
        start = 0
        n = key_s.size
        while start < n:
            end = start + 1
            while end < n and key_s[end] == key_s[start]:
                end += 1
            j = start + int(np.argmax(depth_s[start:end]))
            keep_s[j] = True
            start = end

        yk = y_s[keep_s]
        xk = x_s[keep_s]
        xtk = xt_s[keep_s]

        out[yk, xtk] = left_rgb[yk, xk]
        valid[yk, xtk] = True
    else:
        out[yv, xtv] = left_rgb[yv, xv]
        valid[yv, xtv] = True

    return out, valid


def fill_holes_scanline(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    h, w = valid.shape

    for y in range(h):
        v = valid[y]
        if not np.any(v):
            continue

        left_idx = np.full(w, -1, dtype=np.int32)
        last = -1
        for x in range(w):
            if v[x]:
                last = x
            left_idx[x] = last

        right_idx = np.full(w, -1, dtype=np.int32)
        last = -1
        for x in range(w - 1, -1, -1):
            if v[x]:
                last = x
            right_idx[x] = last

        for x in range(w):
            if v[x]:
                continue
            li = left_idx[x]
            ri = right_idx[x]
            if li == -1 and ri == -1:
                continue
            if li == -1:
                out[y, x] = out[y, ri]
            elif ri == -1:
                out[y, x] = out[y, li]
            else:
                out[y, x] = out[y, li] if (x - li) <= (ri - x) else out[y, x]  # overwritten next line
                out[y, x] = out[y, li] if (x - li) <= (ri - x) else out[y, ri]

    return out


# Backward warp (cv2.remap)

def backward_warp_from_left(left_rgb: np.ndarray, disp_px: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = disp_px.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    x_src = xs + alpha * disp_px
    y_src = ys

    valid = (x_src >= 0) & (x_src <= w - 1) & np.isfinite(disp_px)

    warped = cv2.remap(
        left_rgb,
        x_src.astype(np.float32),
        y_src.astype(np.float32),
        interpolation=BACKWARD_INTERP,
        borderMode=BORDER_MODE,
        borderValue=BORDER_VALUE,
    )

    warped[~valid] = 0
    return warped, valid


def backward_warp_from_right(right_rgb: np.ndarray, disp_px_right: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = disp_px_right.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    x_src = xs - (1.0 - alpha) * disp_px_right
    y_src = ys

    valid = (x_src >= 0) & (x_src <= w - 1) & np.isfinite(disp_px_right)

    warped = cv2.remap(
        right_rgb,
        x_src.astype(np.float32),
        y_src.astype(np.float32),
        interpolation=BACKWARD_INTERP,
        borderMode=BORDER_MODE,
        borderValue=BORDER_VALUE,
    )

    warped[~valid] = 0
    return warped, valid


# Fusion

def fuse_views(vL: np.ndarray, mL: np.ndarray, vR: np.ndarray, mR: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vL, dtype=np.uint8)

    onlyL = mL & ~mR
    onlyR = ~mL & mR
    both = mL & mR

    out[onlyL] = vL[onlyL]
    out[onlyR] = vR[onlyR]

    if np.any(both):
        a = vL[both].astype(np.uint16)
        b = vR[both].astype(np.uint16)
        out[both] = ((a + b) // 2).astype(np.uint8)

    return out


# ============================================================

def main():
    print("=== View Synthesis ===")

    left = load_image_rgb(LEFT_IMAGE_PATH)
    right = load_image_rgb(RIGHT_IMAGE_PATH)

    disp_left_raw = load_disparity(DISP_LEFT_PATH)
    disp_right_raw = load_disparity(DISP_RIGHT_PATH)

    disp_left = rescale_disparity_if_needed(disp_left_raw, DISP_IS_8BIT, DISP_MIN, DISP_MAX)
    disp_right = rescale_disparity_if_needed(disp_right_raw, DISP_IS_8BIT, DISP_MIN, DISP_MAX)

    disp_left = apply_disp_sign(disp_left, DISP_SIGN_LEFT)
    disp_right = apply_disp_sign(disp_right, DISP_SIGN_RIGHT)

    print(f"Left disparity range after rescale+sign:  {disp_left.min():.3f} .. {disp_left.max():.3f}")
    print(f"Right disparity range after rescale+sign: {disp_right.min():.3f} .. {disp_right.max():.3f}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Forward warp (from left)
    fw, fw_mask = forward_warp_from_left(left, disp_left, ALPHA, FORWARD_USE_ZBUFFER)
    if FORWARD_HOLE_FILL:
        fw_filled = fill_holes_scanline(fw, fw_mask)
        save_rgb(out_dir / "view_forward_filled.png", fw_filled)
    save_rgb(out_dir / "view_forward.png", fw)

    # Backward warp from left/right
    bw_l, m_l = backward_warp_from_left(left, disp_left, ALPHA)
    save_rgb(out_dir / "view_backward_left.png", bw_l)

    bw_r, m_r = backward_warp_from_right(right, disp_right, ALPHA)
    save_rgb(out_dir / "view_backward_right.png", bw_r)

    # Fuse
    fused = fuse_views(bw_l, m_l, bw_r, m_r)
    save_rgb(out_dir / "view_fused.png", fused)

    print("Terminé")
    print(f"Résultats dans : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
