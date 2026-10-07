import cv2
import numpy as np

SIZE, PAD = 64, 4


def shape_mask(img_bgr, low_contrast=False, low_thresh=25):
    """Binary mask (255 = shape) from colour distance to the background."""
    img = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    scale = 255 / max(dist.max(), 1)
    d8 = np.clip(dist * scale, 0, 255).astype(np.uint8)
    t, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if low_contrast:   # keep pale shapes even when other shapes are very dark
        thr = min(t / scale, low_thresh)
        mask = ((dist > thr) * 255).astype(np.uint8)
    return mask


def silhouette(path, size=SIZE, pad=PAD):
    img = cv2.imread(str(path))
    mask = shape_mask(img)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [c], -1, 255, -1)
    crop = filled[y:y + h, x:x + w]
    s = max(w, h)
    square = np.zeros((s, s), np.uint8)
    oy, ox = (s - h) // 2, (s - w) // 2
    square[oy:oy + h, ox:ox + w] = crop
    inner = size - 2 * pad
    interp = cv2.INTER_AREA if s > inner else cv2.INTER_LINEAR
    out = cv2.resize(square, (inner, inner), interpolation=interp)
    return cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)