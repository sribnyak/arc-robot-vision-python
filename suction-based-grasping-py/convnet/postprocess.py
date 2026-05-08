"""
Post-process a pixel-wise suction affordance map by removing background and regions
with high variance in 3D surface normals.
"""

import numpy as np
import open3d as o3d
from scipy.ndimage import uniform_filter


def postprocess_affordances(
    affordance_map: np.ndarray,
    input_color: np.ndarray,
    input_depth: np.ndarray,
    background_color: np.ndarray,
    background_depth: np.ndarray,
    camera_intrinsics: np.ndarray,
    color_thresh: float = 0.3,
    depth_thresh: float = 0.02,
    normals_knn: int = 50,
    normals_window: int = 25,
    suction_score_thresh: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Post-process a pixel-wise suction affordance map by removing background
    and regions with high variance in 3D surface normals.

    Args:
        affordance_map: (H, W) float array in range [0, 1]. Modified in-place
            (and also returned).
        input_color: (H, W, 3) RGB float array scaled to [0, 1].
        input_depth: (H, W) depth in meters (float).
        background_color: (H, W, 3) RGB float array scaled to [0, 1].
        background_depth: (H, W) depth in meters (float).
        camera_intrinsics: 3x3 camera intrinsics matrix K, where
            f_x = K[0, 0], f_y = K[1, 1], c_x = K[0, 2], c_y = K[1, 2].
        color_thresh: per-channel absolute difference threshold for color background
            subtraction.
        depth_thresh: depth change threshold (meters) for background subtraction.
        normals_knn: number of neighbors used to estimate normals.
        normals_window: odd int window size for local std filter (e.g., 25).
        suction_score_thresh: minimum suction score to keep affordance
            (values below are zeroed).

    Returns:
        tuple[np.ndarray, np.ndarray]: (affordance_map, surface_normals_map)
        - affordance_map: (H, W) post-processed affordance map (same object modified).
        - surface_normals_map: (H, W, 3) float array of surface normals in camera
          coordinates.
    """

    H, W = input_depth.shape

    # 1. Foreground mask
    fg_color = ~np.all(np.abs(input_color - background_color) < color_thresh, axis=2)
    fg_depth = (background_depth != 0) & (
        np.abs(input_depth - background_depth) > depth_thresh
    )
    fg_mask = fg_color | fg_depth

    # 2. Convert depth to 3D camera coords
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    f_x = camera_intrinsics[0, 0]
    f_y = camera_intrinsics[1, 1]
    c_x = camera_intrinsics[0, 2]
    c_y = camera_intrinsics[1, 2]

    Z = input_depth
    valid = fg_mask & (Z > 0)
    if not np.any(valid):
        # nothing in foreground; zero affordances and return empty normals
        affordance_map = affordance_map.copy()  # TODO is it ok?
        affordance_map[~fg_mask] = 0
        return affordance_map, np.zeros((H, W, 3), dtype=np.float32)

    X = (x_grid - c_x) * Z / f_x
    Y = (y_grid - c_y) * Z / f_y
    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)

    # 3. Estimate normals with Open3D
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # Ensure there are enough points for requested KNN
    search_knn = max(3, min(normals_knn, len(pts) - 1))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=search_knn))
    normals = np.asarray(pcd.normals)  # (N,3)

    # 4. Orient normals toward camera (camera at origin)
    vec_to_cam = -pts  # vector from point to sensor at origin
    dot = np.einsum("ij,ij->i", normals, vec_to_cam)
    normals[dot < 0] = -normals[dot < 0]

    # 5. Reproject normals into image map
    # Compute pixel coordinates for the valid points (use original indices)
    pix_x = np.round((pts[:, 0] * f_x / pts[:, 2]) + c_x).astype(int)
    pix_y = np.round((pts[:, 1] * f_y / pts[:, 2]) + c_y).astype(int)
    # Clamp to image
    pix_x = np.clip(pix_x, 0, W - 1)
    pix_y = np.clip(pix_y, 0, H - 1)

    surface_normals = np.zeros((H, W, 3), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.int32)

    # Accumulate normals for pixels with multiple points then average
    for px, py, n in zip(pix_x, pix_y, normals):
        surface_normals[py, px] += n
        count[py, px] += 1
    nonzero = count > 0
    surface_normals[nonzero] /= count[nonzero][..., None]
    # normalize resulting averaged normals (guard against zeros)
    norms = np.linalg.norm(surface_normals, axis=2, keepdims=True)
    nz = norms > 1e-8
    surface_normals[nz[..., 0]] /= norms[nz]

    # 6. Compute local std per channel via box filter -> then mean across channels
    # uniform_filter works on each channel separately; use size=(window, window, 1)
    w = normals_window
    # Pad/handle edges internally by uniform_filter
    mean = uniform_filter(surface_normals, size=(w, w, 1))
    mean_sq = uniform_filter(surface_normals * surface_normals, size=(w, w, 1))
    std = np.sqrt(np.maximum(0.0, mean_sq - mean * mean))
    mean_std_normals = np.mean(std, axis=2)

    # Avoid division by zero
    denom = mean_std_normals.max()
    if denom <= 0:
        normal_score = np.ones_like(mean_std_normals, dtype=np.float32)
    else:
        normal_score = 1.0 - (mean_std_normals / denom)

    # 7. Apply thresholds: zero-out affordance where normal_score < thresh OR not foreground
    affordance_out = affordance_map.copy()
    affordance_out[(normal_score < suction_score_thresh) | (~fg_mask)] = 0.0

    return affordance_out, surface_normals
