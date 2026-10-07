import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src.preprocess import shape_mask, SIZE, PAD
from src.features import contour_features

MIN_AREA = 150
SPLIT_SOLIDITY = 0.92   # blobs less convex than this are candidates for splitting


def mask_to_silhouette(filled, size=SIZE, pad=PAD):
    ys, xs = np.nonzero(filled)
    crop = filled[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = max(h, w)
    square = np.zeros((s, s), np.uint8)
    oy, ox = (s - h) // 2, (s - w) // 2
    square[oy:oy + h, ox:ox + w] = crop
    inner = size - 2 * pad
    interp = cv2.INTER_AREA if s > inner else cv2.INTER_LINEAR
    out = cv2.resize(square, (inner, inner), interpolation=interp)
    return cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


def split_blob(filled, min_radius=6, min_distance=10):
    from skimage.feature import peak_local_max      # only this older path needs scikit-image
    from skimage.segmentation import watershed
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), 2)
    peaks = peak_local_max(dist, min_distance=min_distance, threshold_abs=min_radius,
                           labels=(filled > 0).astype(int), exclude_border=False)
    if len(peaks) < 2:
        return [filled]
    markers = np.zeros(filled.shape, np.int32)
    for i, (y, x) in enumerate(peaks, 1):
        markers[y, x] = i
    lab = watershed(-dist, markers, mask=filled > 0)
    parts = []
    for i in range(1, len(peaks) + 1):
        m = (lab == i).astype(np.uint8) * 255
        if cv2.countNonZero(m) >= MIN_AREA:
            parts.append(m)
    return parts or [filled]


def _prominent_peaks(dist, h, levels=60):
    """Marker image for the peaks of `dist` that rise at least h above the saddle joining them
    (same idea as h-maxima, using scipy only). A flat ridge counts as one peak."""
    dmax = float(dist.max())
    markers = np.zeros(dist.shape, np.int32)
    peak_level, permanent, rep = {}, set(), {}
    next_id = 1
    for t in np.linspace(dmax, 0.0, levels, endpoint=False):
        if t <= 0:
            break
        lab, n = ndi.label(dist >= t)
        comp_of = {i: lab[rep[i]] for i in peak_level}
        occupied = set(comp_of.values())
        for comp in range(1, n + 1):                       # new peaks
            if comp not in occupied:
                region = lab == comp
                markers[region] = next_id
                peak_level[next_id] = t
                rep[next_id] = tuple(np.argwhere(region)[0])
                comp_of[next_id] = comp
                next_id += 1
        groups = {}
        for i, comp in comp_of.items():
            groups.setdefault(comp, []).append(i)
        for ids in groups.values():                        # merges
            if len(ids) < 2:
                continue
            ids.sort(key=lambda i: -peak_level[i])
            for i in ids[1:]:
                if i in permanent:
                    continue
                if peak_level[i] - t < h:                  # shallow peak: absorb it
                    markers[markers == i] = 0
                    del peak_level[i], rep[i]
                else:
                    permanent.add(i)
    return markers


def _flood(dist, markers, region, levels=60):
    """Marker-based watershed: grow the marker labels from the highest level of `dist`
    downwards, so two regions meet at the saddle between their peaks."""
    lab = markers.copy()
    for t in np.linspace(float(dist.max()), 0.0, levels, endpoint=False):
        allowed = (dist >= t) & region
        while True:
            grown = ndi.grey_dilation(lab, size=(3, 3))
            new = (lab == 0) & allowed & (grown > 0)
            if not new.any():
                break
            lab[new] = grown[new]
    while True:                                   # whatever is left of the blob
        grown = ndi.grey_dilation(lab, size=(3, 3))
        new = (lab == 0) & region & (grown > 0)
        if not new.any():
            break
        lab[new] = grown[new]
    return lab


def split_blob_hmax(filled, min_area):
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(filled))
    pad = 3
    y0, x0 = max(y - pad, 0), max(x - pad, 0)
    y1, x1 = min(y + h + pad, filled.shape[0]), min(x + w + pad, filled.shape[1])
    blob = filled[y0:y1, x0:x1]
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    if dmax < 3:
        return [filled]
    dist = cv2.GaussianBlur(dist, (0, 0), max(1.0, 0.03 * dmax))
    markers = _prominent_peaks(dist, max(1.0, 0.15 * dmax))
    ids = [int(i) for i in np.unique(markers) if i > 0]
    if len(ids) < 2:
        return [filled]
    lab = _flood(dist, markers, blob > 0)
    parts = []
    for i in ids:
        m = np.zeros(filled.shape, np.uint8)
        m[y0:y1, x0:x1][(lab == i) & (blob > 0)] = 255
        if cv2.countNonZero(m) >= min_area:
            parts.append(m)
    return parts or [filled]


def detect_shapes(img_bgr, model, feature_names, min_area=MIN_AREA, split=True, robust=False):
    if robust:
        mask = shape_mask(img_bgr, low_contrast=True)
        min_area = max(min_area, int(1.5e-4 * mask.size))
    else:
        mask = shape_mask(img_bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, feats = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        filled = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(filled, [c], -1, 255, -1)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        parts = [filled]
        if split and hull_area > 0 and area / hull_area < SPLIT_SOLIDITY:
            parts = split_blob_hmax(filled, min_area) if robust else split_blob(filled)
        for p in parts:
            feats.append(contour_features(mask_to_silhouette(p)))
            boxes.append(cv2.boundingRect(cv2.findNonZero(p)))
    if not boxes:
        return []
    labels = model.predict(pd.DataFrame(feats)[feature_names])
    return list(zip(boxes, labels))


def draw_result(img_bgr, results):
    out = img_bgr.copy()
    s = max(1.0, max(out.shape[:2]) / 500)
    t = max(1, int(round(s)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, w, h), label in results:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 160, 0), 2 * t)
        cv2.putText(out, label, (x, max(y - 5, int(14 * s))), font, 0.5 * s, (0, 0, 200), t, cv2.LINE_AA)
    cv2.putText(out, f"count: {len(results)}", (8, int(22 * s)), font, 0.65 * s, (0, 0, 0), 2 * t, cv2.LINE_AA)
    return out