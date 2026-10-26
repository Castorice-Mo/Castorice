
from pathlib import Path

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    p = str(path)
    if p.isascii():
        return cv2.imread(p, flags)
    try:
        data = np.fromfile(p, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, img, params=None):
    if img is None:
        return False
    p = str(path)
    params = params or []
    if p.isascii():
        return bool(cv2.imwrite(p, img, params))
    ext = Path(p).suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(p)
    return True
