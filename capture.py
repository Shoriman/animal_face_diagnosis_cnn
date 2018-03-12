import argparse
import time

import cv2

parser = argparse.ArgumentParser(description='Animal face diagnosis')
parser.add_argument('--face_cascade', '-f',
                    default='haarcascade_frontalface_default.xml')
parser.add_argument('--eye_cascade', '-e',
                    default='haarcascade_eye.xml')
args = parser.parse_args()

face_cascade = cv2.CascadeClassifier(args.face_cascade)
eye_cascade = cv2.CascadeClassifier(args.eye_cascade)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detected = False
while(True):
    ret, frame = cap.read()
    if ret is False:
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        if len(faces) > 0:
            for j, (x, y, w, h) in enumerate(faces):
                if h/frame.shape[0] < 0.1 or w/frame.shape[1] < 0.1:
                    continue
                pw, ph = int(w*0.05), int(h*0.05)
                roi = frame[y-ph:y+h+ph, x-pw:x+w+pw]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray_roi)
                if len(eyes) < 2:
                    continue
                # TODO: 判定処理など
                detected = True
            if detected:
                break
        else:
            continue
cap.release()
# TODO: 結果画像を出力
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
