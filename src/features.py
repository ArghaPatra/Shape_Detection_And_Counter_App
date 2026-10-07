import cv2
import numpy as np


def contour_features(sil):
    mask = (sil > 127).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    (_, _), r = cv2.minEnclosingCircle(c)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    f = {
        "area": area,
        "perimeter": peri,
        "circularity": 4 * np.pi * area / peri ** 2,
        "solidity": area / hull_area,
        "circle_fill": area / (np.pi * r ** 2),
        "rect_fill": area / (rw * rh),
        "rect_aspect": max(rw, rh) / max(min(rw, rh), 1e-6),
        "n_vertices_2": len(cv2.approxPolyDP(c, 0.02 * peri, True)),
        "n_vertices_4": len(cv2.approxPolyDP(c, 0.04 * peri, True)),
    }
    for i, v in enumerate(hu):
        f[f"hu{i + 1}"] = v
    return f