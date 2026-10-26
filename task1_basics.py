from pathlib import Path
import cv2

from io_utils import imread, imwrite

BASE = Path(__file__).resolve().parents[1]
IMG_DIR = BASE / "data" / "images"
OUT_DIR = BASE / "outputs" / "task1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_NAMES = ["01.JPG", "02.JPG", "03.JPG"]

images = [IMG_DIR / n for n in IMAGE_NAMES]
missing = [x.name for x in images if not x.exists()]
if missing:
    raise SystemExit(f"缺少图片: {missing}  (目录: {IMG_DIR})")

for path in images:
    img = imread(str(path))
    if img is None:
        print(f"[跳过] 读取失败: {path.name}")
        continue

    h, w, c = img.shape
    print(f"\n{path.name}: 宽={w} 高={h} 通道={c}")
    # print(f"  像素数={w * h}  像素值个数={w * h * c}  数据类型={img.dtype}")

    stem = path.stem
    small = cv2.resize(img, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)   #缩放
    crop = img[h // 4:h * 3 // 4, w // 4:w * 3 // 4]     #裁剪
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #灰度图

    for tag, out_img in [("resized", small), ("crop", crop), ("gray", gray)]:
      out_path = OUT_DIR / f"{stem}_{tag}.png"
      ok = imwrite(str(out_path), out_img)

