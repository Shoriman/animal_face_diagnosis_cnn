import os
from glob import glob
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('src_images')
parser.add_argument('dataset_dir')
parser.add_argument(
    '--face_cascade_path',
    '-f',
    default='haarcascade_frontalface_default.xml'
)
parser.add_argument(
    '--eye_cascade_path',
    '-e',
    default='haarcascade_eye.xml'
)
args = parser.parse_args()

src_path = args.src_images.rstrip('/')
cls = os.path.basename(src_path)
dst_dir = os.path.join(args.dataset_dir, cls)
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
face_cascade = cv2.CascadeClassifier(args.face_cascade_path)
eye_cascade = cv2.CascadeClassifier(args.eye_cascade_path)
dst_size = 100

dirs = [x for x in glob(args.src_images+'/*') if os.path.isdir(x)]
for d in dirs:
    who = os.path.basename(d)
    img_paths = glob(d+'/*.jpeg') + glob(d+'/*.png') + glob(d+'/*.tiff')
    for i, p in enumerate(img_paths):
        _, ext = os.path.splitext(p)
        img = cv2.imread(p)
        if img.dtype.type is not np.uint8:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        for j, (x, y, w, h) in enumerate(faces):
            # 検出領域が元画像に対して非常に小さい場合を除く
            if h/img.shape[0] < 0.1 or w/img.shape[1] < 0.1:
                continue
            # 実際の検出領域よりも少し大きくROIをとる
            pw, ph = int(w*0.05), int(h*0.05)
            roi = img[y-ph:y+h+ph, x-pw:x+w+pw]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 瞳領域が2つ未満の画像は除く
            eyes = eye_cascade.detectMultiScale(gray_roi)
            if len(eyes) < 2:
                continue
            # 画像をresize
            if roi.shape[0] > dst_size:
                method = cv2.INTER_AREA
            else:
                method = cv2.INTER_CUBIC
            roi = cv2.resize(roi, (dst_size, dst_size), interpolation=method)
            # 画像を保存
            dst_path = os.path.join(dst_dir,
                                    '{}_{}_{}{}'.format(who, i, j, ext))
            cv2.imwrite(dst_path, roi)
