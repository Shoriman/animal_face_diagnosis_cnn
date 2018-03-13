# -*- coding: utf-8 -*-

import argparse
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from chainer import serializers

from src.models.discriminator import discriminator

mpl.rcParams['font.family'] = 'AppleGothic'

parser = argparse.ArgumentParser(description='Animal face diagnosis')
parser.add_argument('--face_cascade', '-f',
                    default='haarcascade_frontalface_default.xml')
parser.add_argument('--eye_cascade', '-e',
                    default='haarcascade_eye.xml')
parser.add_argument('--model', '-m', default='model_files/discriminator.npz')
parser.add_argument('--arch', '-a', default='model_files/arch.json')
parser.add_argument('--param', '-p', default='model_files/param.json')
parser.add_argument('--mean_img', default='model_files/mean_img.npy')
parser.add_argument('--background', default='images/back.jpeg')
parser.add_argument('--inu', default='images/inu.jpg')
parser.add_argument('--neko', default='images/neko.jpg')
parser.add_argument('--tanuki', default='images/tanuki.jpg')
parser.add_argument('--kitsune', default='images/kitsune.jpg')
parser.add_argument('--rakuda', default='images/rakuda.jpg')
parser.add_argument('--kawauso', default='images/kawauso.jpg')
args = parser.parse_args()

insize = (100, 100)
roi_size = (200, 200)

face_cascade = cv2.CascadeClassifier(args.face_cascade)
eye_cascade = cv2.CascadeClassifier(args.eye_cascade)
with open(args.arch, 'r') as fp:
    arch = json.load(fp)
with open(args.param, 'r') as fp:
    param = json.load(fp)
model = discriminator(arch, dr=param['dr'], bn=param['bn'])
serializers.load_npz(args.model, model)
model.train = False
mean_img = np.load(args.mean_img)

animals = {
    0: cv2.resize(cv2.imread(args.inu), roi_size),
    1: cv2.resize(cv2.imread(args.kawauso), roi_size),
    2: cv2.resize(cv2.imread(args.kitsune), roi_size),
    3: cv2.resize(cv2.imread(args.neko), roi_size),
    4: cv2.resize(cv2.imread(args.rakuda), roi_size),
    5: cv2.resize(cv2.imread(args.tanuki), roi_size),
}
names = {
    0: 'イヌ',
    1: 'カワウソ',
    2: 'キツネ',
    3: 'ネコ',
    4: 'ラクダ',
    5: 'タヌキ',
}

cap = cv2.VideoCapture(0)

detected = False
while(True):
    ret, frame = cap.read()
    if ret is False:
        break
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k & 0xFF == ord('e'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        if len(faces) >= 1:
            roi_lst = []
            in_lst = []
            for j, (x, y, w, h) in enumerate(faces):
                if h/frame.shape[0] < 0.1 or w/frame.shape[1] < 0.1:
                    continue
                pw, ph = int(w*0.05), int(h*0.05)
                roi = frame[y-ph:y+h+ph, x-pw:x+w+pw]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray_roi)
                if len(eyes) < 2:
                    continue
                roi_lst.append(cv2.resize(roi, roi_size))
                in_lst.append(cv2.resize(roi, insize))
            if len(roi_lst) >= 1:
                in_array = np.array(in_lst).astype('f') / 255
                in_array -= mean_img
                in_array = in_array.transpose(0, 3, 1, 2)
                prob = model(in_array).data
                pred = np.argmax(prob, axis=1)

                myround = lambda x: int((x*2+1)//2)
                row = myround(len(pred)/2)
                if len(roi_lst) == 1:
                    col = 1
                else:
                    col = 2
                region = row * 100 + col * 10
                plt.figure(figsize=(8, 4), num='判別結果')
                for i, p in enumerate(pred):
                    plt.subplot(region+i+1)
                    res_img = cv2.hconcat([roi_lst[i], animals[p]])
                    plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.title('{}人目: {}'.format(i+1, names[p]),
                              fontsize=28)
                plt.tight_layout()
                plt.show()
            continue
        else:
            continue

    if k & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
