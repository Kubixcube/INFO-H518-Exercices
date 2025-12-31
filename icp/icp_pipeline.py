import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

# =========================
# CONFIG
# =========================
LEFT_IMAGE_PATH = "cone0.png"
RIGHT_IMAGE_PATH = "cone1.png"
DISPARITY_8BIT_PATH = "cones_disparity_occlusion_filled_smoothed.png"  # disparity left->right (8-bit)

# IPOL mapping
DISP_MIN = -51
DISP_MAX = -17

# Sampling
PIXEL_STRIDE = 2
DISP_EPS = 1e-3

# "Calib" minimale (peut rester relative si tu n'as pas mieux)
# Important: la profondeur est Z = FX * BASELINE / d
FX = 700.0
FY = 700.0
BASELINE = 0.12  # unités cohérentes avec Z (mètres conseillé)

# Filtrage profondeur (évite les outliers extrêmes)
Z_MIN = 0.01
Z_MAX = 500.0

# Prétraitement / ICP
VOXEL_SCALES = [0.05, 0.02, 0.01]      # multi-échelle
MAX_CORR_FACTORS = [3.0, 2.5, 2.0]     # max_corr = voxel * factor
ICP_ITERS = [60, 60, 80]

# Sorties
OUTPUT_DIR = "out_icp_lr"
OUT_LEFT = "cloud_left_cam.ply"
OUT_RIGHT = "cloud_right_cam.ply"
OUT_RIGHT_ALIGNED = "cloud_right_aligned_to_left.ply"
OUT_MERGED = "cloud_merged.ply"
OUT_T = "T_right_to_left.txt"

# Debug / visualisation
SHOW_WINDOWS = True  # mettre False si tu veux tout en batch


# =========================
# IO
# =========================
def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def decode_disp(d8: np.ndarray) -> np.ndarray:
    """
    Mappe l'image 8-bit vers la plage [DISP_MIN, DISP_MAX].
    """
    return (d8.astype(np.float32) / 255.0) * (DISP_MAX - DISP_MIN) + DISP_MIN


# =========================
# Point clouds
# =========================
def to_o3d(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def build_pointclouds_from_disp_lr(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    disp8: np.ndarray,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Construit deux nuages:
      - nuage L dans repère caméra gauche
      - nuage R dans repère caméra droite (utilise uR)

    Hypothèse: disp est left->right au sens uR = uL - d, avec d > 0.
    Or ta plage IPOL étant négative, on inverse le signe pour obtenir d > 0.
    """
    h, w = disp8.shape

    disp_raw = decode_disp(disp8)          # typiquement négatif chez toi
    disp = (-disp_raw).astype(np.float32)  # maintenant d > 0 attendu (~[17..51])

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    pts_L, col_L = [], []
    pts_R, col_R = [], []

    for v in range(0, h, PIXEL_STRIDE):
        for uL in range(0, w, PIXEL_STRIDE):
            d = float(disp[v, uL])
            if d < DISP_EPS:
                continue

            uR = uL - d
            if uR < 0.0 or uR >= (w - 1.001):
                continue

            Z = (FX * BASELINE) / (d + DISP_EPS)
            if Z < Z_MIN or Z > Z_MAX:
                continue

            # Repère gauche
            XL = (uL - cx) * Z / FX
            YL = -(v - cy) * Z / FY

            # Repère droit (en utilisant uR)
            XR = (uR - cx) * Z / FX
            YR = -(v - cy) * Z / FY

            # Couleur gauche (nearest, uL entier)
            bL, gL, rL = left_bgr[v, uL]
            cL = np.array([rL, gL, bL], dtype=np.float32) / 255.0

            # Couleur droite (bilinéaire 1D sur u, v entier)
            u0 = int(np.floor(uR))
            u1 = u0 + 1
            a = float(uR - u0)

            c0 = right_rgb[v, u0].astype(np.float32)
            c1 = right_rgb[v, u1].astype(np.float32)
            cR = ((1.0 - a) * c0 + a * c1) / 255.0

            pts_L.append([XL, YL, Z])
            col_L.append(cL)

            pts_R.append([XR, YR, Z])
            col_R.append(cR)

    pts_L = np.asarray(pts_L, dtype=np.float32)
    col_L = np.asarray(col_L, dtype=np.float32)
    pts_R = np.asarray(pts_R, dtype=np.float32)
    col_R = np.asarray(col_R, dtype=np.float32)

    return to_o3d(pts_L, col_L), to_o3d(pts_R, col_R)


# =========================
# Preprocess / ICP helpers
# =========================
def preprocess_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    p = pcd.voxel_down_sample(voxel_size)

    if len(p.points) == 0:
        return p

    p, _ = p.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    if len(p.points) == 0:
        return p

    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3.0,
            max_nn=30,
        )
    )
    return p


def run_icp_point_to_plane(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    max_corr: float,
    max_iter: int,
) -> o3d.pipelines.registration.RegistrationResult:
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_corr,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )


def multiscale_icp_point_to_plane(
    source_raw: o3d.geometry.PointCloud,
    target_raw: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    voxel_sizes: list[float],
    max_corr_factors: list[float],
    iters: list[int],
) -> np.ndarray:
    if not (len(voxel_sizes) == len(max_corr_factors) == len(iters)):
        raise ValueError("voxel_sizes, max_corr_factors, iters must have same length")

    T = init_T.copy()

    for vs, f, niter in zip(voxel_sizes, max_corr_factors, iters):
        src = preprocess_pcd(source_raw, voxel_size=vs)
        tgt = preprocess_pcd(target_raw, voxel_size=vs)

        if len(src.points) == 0 or len(tgt.points) == 0:
            raise RuntimeError("Empty point cloud after preprocessing; adjust parameters")

        max_corr = vs * f
        res = run_icp_point_to_plane(src, tgt, T, max_corr=max_corr, max_iter=niter)
        T = res.transformation

        print(
            f"[ICP scale voxel={vs:.4f}] fitness={res.fitness:.4f} "
            f"rmse={res.inlier_rmse:.6f} max_corr={max_corr:.4f}"
        )

    return T


def pick_best(res_a, res_b):
    # priorité: fitness, puis rmse
    if res_a.fitness > res_b.fitness:
        return res_a
    if res_b.fitness > res_a.fitness:
        return res_b
    return res_a if res_a.inlier_rmse <= res_b.inlier_rmse else res_b


def show_clouds(pcds, title: str):
    if not SHOW_WINDOWS:
        return
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([*pcds, axes], window_name=title)


# =========================
# Main
# =========================
def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    left = read_bgr(LEFT_IMAGE_PATH)
    right = read_bgr(RIGHT_IMAGE_PATH)
    disp8 = read_gray(DISPARITY_8BIT_PATH)

    print("Building point clouds...")
    pcdL, pcdR = build_pointclouds_from_disp_lr(left, right, disp8)

    o3d.io.write_point_cloud(str(out / OUT_LEFT), pcdL)
    o3d.io.write_point_cloud(str(out / OUT_RIGHT), pcdR)
    print("Saved raw:")
    print(" -", (out / OUT_LEFT).resolve())
    print(" -", (out / OUT_RIGHT).resolve())
    print(f"Nb points: left={len(pcdL.points)} right={len(pcdR.points)}")

    show_clouds([pcdL, pcdR], "Raw clouds (Left + Right)")

    # ICP init: +B and -B, then multi-scale refine
    init_plus = np.eye(4, dtype=np.float64)
    init_plus[0, 3] = +BASELINE

    init_minus = np.eye(4, dtype=np.float64)
    init_minus[0, 3] = -BASELINE

    print("\nRunning multi-scale ICP (Right -> Left), init +B")
    T_plus = multiscale_icp_point_to_plane(
        source_raw=pcdR,
        target_raw=pcdL,
        init_T=init_plus,
        voxel_sizes=VOXEL_SCALES,
        max_corr_factors=MAX_CORR_FACTORS,
        iters=ICP_ITERS,
    )

    print("\nRunning multi-scale ICP (Right -> Left), init -B")
    T_minus = multiscale_icp_point_to_plane(
        source_raw=pcdR,
        target_raw=pcdL,
        init_T=init_minus,
        voxel_sizes=VOXEL_SCALES,
        max_corr_factors=MAX_CORR_FACTORS,
        iters=ICP_ITERS,
    )

    # Evaluate both at finest scale
    vs_f = VOXEL_SCALES[-1]
    max_corr_f = vs_f * MAX_CORR_FACTORS[-1]
    src_f = preprocess_pcd(pcdR, voxel_size=vs_f)
    tgt_f = preprocess_pcd(pcdL, voxel_size=vs_f)

    res_plus = run_icp_point_to_plane(src_f, tgt_f, T_plus, max_corr=max_corr_f, max_iter=80)
    res_minus = run_icp_point_to_plane(src_f, tgt_f, T_minus, max_corr=max_corr_f, max_iter=80)
    best = pick_best(res_plus, res_minus)

    print("\nFinal evaluation (finest scale):")
    print(f" +B: fitness={res_plus.fitness:.4f}, rmse={res_plus.inlier_rmse:.6f}")
    print(f" -B: fitness={res_minus.fitness:.4f}, rmse={res_minus.inlier_rmse:.6f}")
    print("\nBEST transformation (Right -> Left):")
    print(best.transformation)

    T_best = best.transformation
    np.savetxt(str(out / OUT_T), T_best, fmt="%.10f")
    print("Saved transform:")
    print(" -", (out / OUT_T).resolve())

    # Apply transform to full-resolution Right cloud
    pcdR_aligned = o3d.geometry.PointCloud(pcdR)
    pcdR_aligned.transform(T_best)
    o3d.io.write_point_cloud(str(out / OUT_RIGHT_ALIGNED), pcdR_aligned)
    print("Saved aligned right:")
    print(" -", (out / OUT_RIGHT_ALIGNED).resolve())

    # Merge (optionnel mais très utile pour la suite + rapport)
    merged = pcdL + pcdR_aligned
    merged = merged.voxel_down_sample(VOXEL_SCALES[-1])
    o3d.io.write_point_cloud(str(out / OUT_MERGED), merged)
    print("Saved merged cloud:")
    print(" -", (out / OUT_MERGED).resolve())

    # Visualisation finale (downsample pour être fluide)
    pcdL_vis = preprocess_pcd(pcdL, voxel_size=VOXEL_SCALES[-1])
    pcdR_vis = preprocess_pcd(pcdR_aligned, voxel_size=VOXEL_SCALES[-1])
    show_clouds([pcdL_vis, pcdR_vis], "After ICP (Right aligned to Left)")

    print("\nDone.")


if __name__ == "__main__":
    main()
