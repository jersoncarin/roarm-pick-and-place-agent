from __future__ import annotations
import cv2
import numpy as np

CAMERA_INDEX = 2  # fallback only if no env loaded
CAMERA_SIZE = (1280, 720)
PREVIEW_SIZE = (1280, 720)
INFER_SIZE = (644, 434)
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50
# TCP offset in the ArUco marker's local axes:
# X follows the marker's left-to-right edge, Y follows top-to-bottom.
CALIB_TCP_X = -120
CALIB_TCP_Y = 15

_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
_ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
_ARUCO_DETECTOR = (
    cv2.aruco.ArucoDetector(_ARUCO_DICTIONARY, _ARUCO_PARAMETERS)
    if hasattr(cv2.aruco, "ArucoDetector")
    else None
)


def open_camera(camera_index: int | None = None) -> cv2.VideoCapture:
    idx = camera_index if camera_index is not None else CAMERA_INDEX
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open USB camera at index {idx}.")
    return cap


def resize_for_preview(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, PREVIEW_SIZE)


def resize_for_inference(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, INFER_SIZE)


def _detect_aruco_markers(
    frame: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    if _ARUCO_DETECTOR is not None:
        corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            _ARUCO_DICTIONARY,
            parameters=_ARUCO_PARAMETERS,
        )

    return corners, ids


def _get_primary_aruco_marker(
    corners: list[np.ndarray],
    ids: np.ndarray | None,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    if ids is None or not corners:
        return None, None

    marker_areas = [
        cv2.contourArea(np.asarray(
            marker_corners, dtype=np.float32).reshape(-1, 2))
        for marker_corners in corners
    ]
    marker_index = int(np.argmax(marker_areas))

    marker_corners = np.asarray(
        corners[marker_index],
        dtype=np.float32,
    ).reshape(-1, 2)
    marker_id = int(ids[marker_index][0])

    return marker_corners, marker_id


def get_aruco_marker_point(frame: np.ndarray) -> tuple[int, int] | None:
    corners, ids = _detect_aruco_markers(frame)
    marker_corners, _ = _get_primary_aruco_marker(corners, ids)

    if marker_corners is None:
        return None

    center = _get_marker_center(marker_corners)
    return int(round(center[0])), int(round(center[1]))


def _get_marker_center(marker_corners: np.ndarray) -> np.ndarray:
    return marker_corners.mean(axis=0)


def _get_marker_axes(marker_corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top_left, top_right, bottom_right, bottom_left = marker_corners

    x_axis = ((top_right - top_left) + (bottom_right - bottom_left)) / 2.0
    y_axis = ((bottom_left - top_left) + (bottom_right - top_right)) / 2.0

    x_norm = np.linalg.norm(x_axis)
    y_norm = np.linalg.norm(y_axis)

    if x_norm <= 1e-6 or y_norm <= 1e-6:
        return (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        )

    return x_axis / x_norm, y_axis / y_norm


def _get_calibrated_tcp_point(marker_corners: np.ndarray) -> tuple[int, int]:
    center = _get_marker_center(marker_corners)
    x_axis, y_axis = _get_marker_axes(marker_corners)
    tcp_point = center + x_axis * CALIB_TCP_X + y_axis * CALIB_TCP_Y

    return int(round(tcp_point[0])), int(round(tcp_point[1]))


def get_aruco_tcp_point(frame: np.ndarray) -> tuple[int, int] | None:
    corners, ids = _detect_aruco_markers(frame)
    marker_corners, _ = _get_primary_aruco_marker(corners, ids)

    if marker_corners is None:
        return None

    return _get_calibrated_tcp_point(marker_corners)


def draw_aruco_marker_point(frame: np.ndarray) -> np.ndarray:
    result = frame.copy()
    corners, ids = _detect_aruco_markers(frame)
    marker_corners, marker_id = _get_primary_aruco_marker(corners, ids)

    if marker_corners is None or marker_id is None:
        return result

    cv2.aruco.drawDetectedMarkers(result, corners, ids)

    center = _get_marker_center(marker_corners)
    center_point = int(round(center[0])), int(round(center[1]))
    tcp_point = _get_calibrated_tcp_point(marker_corners)

    cv2.circle(result, center_point, 4, (0, 0, 255), -1)
    cv2.line(result, center_point, tcp_point, (0, 255, 255), 2)
    cv2.circle(result, tcp_point, 4, (255, 0, 0), -1)

    return result


def detect_and_draw_aruco(
    frame: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Single ArUco detection pass — returns (annotated_frame, tcp_point)."""
    corners, ids = _detect_aruco_markers(frame)
    marker_corners, marker_id = _get_primary_aruco_marker(corners, ids)

    if marker_corners is None or marker_id is None:
        return frame, None

    result = frame.copy()
    cv2.aruco.drawDetectedMarkers(result, corners, ids)

    center = _get_marker_center(marker_corners)
    center_point = int(round(center[0])), int(round(center[1]))
    tcp_point = _get_calibrated_tcp_point(marker_corners)

    cv2.circle(result, center_point, 4, (0, 0, 255), -1)
    cv2.line(result, center_point, tcp_point, (0, 255, 255), 2)
    cv2.circle(result, tcp_point, 4, (255, 0, 0), -1)

    return result, tcp_point


def _elongated_end_points(
    mask_bin: np.ndarray,
    contour: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return (handle_point, head_point) in mask coordinates.

    handle = narrower end (less mask density), head = wider/bulkier end.
    Uses boxPoints corners directly — no angle math.
    """
    if len(contour) < 5:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float64)

    d01 = np.linalg.norm(box[0] - box[1])
    d12 = np.linalg.norm(box[1] - box[2])

    if d01 >= d12:
        end1 = (box[1] + box[2]) / 2.0
        end2 = (box[0] + box[3]) / 2.0
    else:
        end1 = (box[0] + box[1]) / 2.0
        end2 = (box[2] + box[3]) / 2.0

    short_side = min(d01, d12)
    radius = max(int(short_side * 0.5), 4)
    img_h, img_w = mask_bin.shape

    def _density(pt):
        x, y = int(round(pt[0])), int(round(pt[1]))
        y1 = max(0, y - radius)
        y2 = min(img_h, y + radius + 1)
        x1 = max(0, x - radius)
        x2 = min(img_w, x + radius + 1)
        region = mask_bin[y1:y2, x1:x2]
        return int(region.sum()) if region.size > 0 else 0

    d1, d2 = _density(end1), _density(end2)
    e1 = (float(end1[0]), float(end1[1]))
    e2 = (float(end2[0]), float(end2[1]))
    if d1 <= d2:
        return e1, e2
    return e2, e1


def anchor_point_from_mask(
    mask,
    anchor: str,
    target_size: tuple[int, int],
) -> tuple[int, int] | None:
    """Compute an anchor/grasp point from a segmentation mask, scaled to target_size.

    The returned point is ALWAYS inside the binary mask (mask==1).
    """
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr[0]
    elif mask_arr.ndim == 3 and mask_arr.shape[2] == 1:
        mask_arr = mask_arr[:, :, 0]
    if mask_arr.ndim != 2:
        return None

    mask_bin = (mask_arr > 0).astype(np.uint8)
    mask_bin = cv2.resize(
        mask_bin, INFER_SIZE, interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None

    mask_pts = np.argwhere(mask_bin > 0)
    if len(mask_pts) == 0:
        return None

    def _snap(px: float, py: float) -> tuple[float, float]:
        """If (px, py) is outside the mask, snap to the nearest mask pixel."""
        ix = int(round(px))
        iy = int(round(py))
        h, w = mask_bin.shape
        if 0 <= iy < h and 0 <= ix < w and mask_bin[iy, ix] > 0:
            return px, py
        dists = (mask_pts[:, 1].astype(np.float64) - px) ** 2 + \
                (mask_pts[:, 0].astype(np.float64) - py) ** 2
        nearest = mask_pts[np.argmin(dists)]
        return float(nearest[1]), float(nearest[0])

    sx = target_size[0] / INFER_SIZE[0]
    sy = target_size[1] / INFER_SIZE[1]

    def _to_target(px: float, py: float) -> tuple[int, int]:
        px, py = _snap(px, py)
        tx = int(round(max(0.0, min(px * sx, target_size[0] - 1))))
        ty = int(round(max(0.0, min(py * sy, target_size[1] - 1))))
        return tx, ty

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    if anchor == "center":
        return _to_target(cx, cy)

    if anchor == "narrow":
        best_row = -1
        best_width = float("inf")
        best_mid_x = cx
        h, w = mask_bin.shape
        for row in range(h):
            cols = np.where(mask_bin[row] > 0)[0]
            if len(cols) < 2:
                continue
            width = float(cols[-1] - cols[0] + 1)
            if width < best_width:
                best_width = width
                best_row = row
                best_mid_x = float(cols[0] + cols[-1]) / 2.0
        if best_row >= 0:
            return _to_target(best_mid_x, float(best_row))
        return _to_target(cx, cy)

    if anchor in ("handle", "head"):
        ends = _elongated_end_points(mask_bin, contour)
        if ends is None:
            return _to_target(cx, cy)
        handle_pt, head_pt = ends
        pt = handle_pt if anchor == "handle" else head_pt
        return _to_target(pt[0], pt[1])

    left_pt = contour[contour[:, :, 0].argmin()][0]
    right_pt = contour[contour[:, :, 0].argmax()][0]
    top_pt = contour[contour[:, :, 1].argmin()][0]
    bottom_pt = contour[contour[:, :, 1].argmax()][0]

    if anchor == "top":
        return _to_target(cx, top_pt[1])
    elif anchor == "bottom":
        return _to_target(cx, bottom_pt[1])
    elif anchor == "left":
        return _to_target(left_pt[0], cy)
    elif anchor == "right":
        return _to_target(right_pt[0], cy)
    elif anchor == "top_left":
        return _to_target(left_pt[0], top_pt[1])
    elif anchor == "top_right":
        return _to_target(right_pt[0], top_pt[1])
    elif anchor == "bottom_left":
        return _to_target(left_pt[0], bottom_pt[1])
    elif anchor == "bottom_right":
        return _to_target(right_pt[0], bottom_pt[1])

    return _to_target(cx, cy)


def scale_bbox_from_inference(
    bbox: list[float],
    target_size: tuple[int, int],
) -> list[float]:
    x_scale = target_size[0] / INFER_SIZE[0]
    y_scale = target_size[1] / INFER_SIZE[1]

    return [
        bbox[0] * x_scale,
        bbox[1] * y_scale,
        bbox[2] * x_scale,
        bbox[3] * y_scale,
    ]


def draw_detections_on_preview(
    frame: np.ndarray,
    detections: dict[str, tuple[list[list[float]], list[np.ndarray], list[float]]],
    color=(0, 255, 0),
    alpha: float = 0.35,
    show_labels: bool = True,
) -> np.ndarray:
    result = frame.copy()
    preview_w, preview_h = PREVIEW_SIZE

    for object_name, (bboxes, masks, scores) in detections.items():
        count = max(len(bboxes), len(masks), len(scores))

        for i in range(count):
            mask = masks[i] if i < len(masks) else None
            score = scores[i] if i < len(scores) else None

            if mask is None:
                continue

            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
                mask_arr = mask_arr[0]
            elif mask_arr.ndim == 3 and mask_arr.shape[2] == 1:
                mask_arr = mask_arr[:, :, 0]
            if mask_arr.ndim != 2:
                continue

            mask_bin = (mask_arr > 0).astype(np.uint8)
            mask_resized = cv2.resize(
                mask_bin,
                (preview_w, preview_h),
                interpolation=cv2.INTER_NEAREST,
            )

            mask_bool = mask_resized.astype(bool)
            result[mask_bool] = (
                result[mask_bool].astype(np.float32) * (1.0 - alpha)
                + np.array(color, dtype=np.float32) * alpha
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(result, contours, -1, color, 1)

            if show_labels and contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)

                display_name = object_name if count == 1 else f"{object_name}#{i+1}"
                if score is not None:
                    score_pct = int(np.ceil(score * 100))
                    label = f"{display_name} {score_pct}%"
                else:
                    label = display_name

                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                text_x = x
                text_y = y - 8
                if text_y - text_h < 0:
                    text_y = y + text_h + 8

                cv2.rectangle(
                    result,
                    (text_x, text_y - text_h - baseline),
                    (text_x + text_w, text_y + baseline),
                    color, -1)
                cv2.putText(
                    result, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return result


def draw_fps_overlay(
    frame: np.ndarray,
    fps: float,
    margin: tuple[int, int] = (10, 25),
    font_scale: float = 0.7,
    thickness: int = 2,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    result = frame.copy()
    label = f"FPS: {fps:.1f}"
    (text_width, _), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    x = max(0, result.shape[1] - text_width - margin[0])
    y = margin[1]
    cv2.putText(
        result,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return result


def update_smoothed_fps(
    current_fps: float,
    dt: float,
    alpha: float = 0.1,
) -> float:
    if dt <= 0:
        return current_fps

    instant_fps = 1.0 / dt
    keep_weight = 1.0 - alpha
    return current_fps * keep_weight + instant_fps * alpha
