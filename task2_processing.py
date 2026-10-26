from pathlib import Path
import cv2
import numpy as np

from io_utils import imread, imwrite

BASE = Path(__file__).resolve().parents[1]
IMG_DIR = BASE / "data" / "images"
OUT_DIR = BASE / "outputs" / "task2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_NAME = "02.JPG"
MAX_SIDE = 800

img = imread(str(IMG_DIR / IMAGE_NAME))
if img is None:
    raise SystemExit(f"读取失败: {IMG_DIR / IMAGE_NAME}")

h, w = img.shape[:2]
scale = MAX_SIDE / max(h, w)
if scale < 1:
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
print("处理尺寸:", img.shape)


def save(name, out_img):
    path = OUT_DIR / f"{name}.png"
    ok = imwrite(str(path), out_img)
    white_ratio = float((out_img > 127).mean() * 100)
    print(f"  {name:22s} 保存={ok}  白色占比={white_ratio:5.1f}%")
    return ok

#灰度图 && 高斯滤波
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gray, (5, 5), 0)

save("01_gray", gray)
save("02_gauss_5x5", gauss)

#阈值分割
_, th_80 = cv2.threshold(gauss, 80, 255, cv2.THRESH_BINARY)
_, th_180 = cv2.threshold(gauss, 180, 255, cv2.THRESH_BINARY)
_, th_otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(gauss, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 31, 5)

save("03_thresh_80", th_80)
save("04_thresh_180", th_180)
save("05_otsu", th_otsu)
save("06_adaptive", adaptive)

#边缘图
edge_lo = cv2.Canny(gauss, 30, 100)  # 低阈值：边缘多，含噪点
edge_hi = cv2.Canny(gauss, 100, 200)  # 高阈值：边缘少，但可能断

save("07_canny_30_100", edge_lo)
save("08_canny_100_200", edge_hi)

#形态学处理
k3 = np.ones((3, 3), np.uint8)
k7 = np.ones((7, 7), np.uint8)

eroded = cv2.erode(th_otsu, k3, iterations=1)
dilated = cv2.dilate(th_otsu, k3, iterations=1)
opened = cv2.morphologyEx(th_otsu, cv2.MORPH_OPEN, k7)  # 先腐蚀后膨胀：去小白点
closed = cv2.morphologyEx(th_otsu, cv2.MORPH_CLOSE, k7)  # 先膨胀后腐蚀：补小黑洞

save("09_erode_3x3", eroded)
save("10_dilate_3x3", dilated)
save("11_open_7x7", opened)
save("12_close_7x7", closed)

