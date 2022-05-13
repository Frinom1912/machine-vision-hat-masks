import os

import cv2
import mediapipe as mp
import time
from math import hypot
import numpy as np


# Root to images dir, note that images MUST be png with transparent background
# Image size doesnt matter as it is auto-scaled
ROOT = "some_dir\\Hats"


class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.controlPoints = []

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon, min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        self.controlPoints = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # Uncomment this to find your points
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #           0.7, (0, 255, 0), 1)

                    face.append([x, y])
                faces.append(face)
                for face in faces:
                    # 54 and 284 are the points of forehead, you can find your points via cv2.putText up
                    self.controlPoints.append([face[54], face[284]])
        return img

    # Images still being black after overlay, open Merge request if you managed to solve this
    def transparentOverlay(self, img, overlayImage):
        if not len(self.controlPoints):
            return img
        h, w, _ = overlayImage.shape  # Size of foreground
        rows, cols, _ = img.shape  # Size of background Image
        unparsedXLeft, unparsedYLeft = self.controlPoints[0][0][0], self.controlPoints[0][0][1]
        xLeft, yLeft = int(unparsedXLeft), int(unparsedYLeft)
        yLeft -=20
        xLeft -= 50
        unparsedXRight, unparsedYRight = self.controlPoints[0][1][0], self.controlPoints[0][1][1]
        xRight, yRight = int(unparsedXRight), int(unparsedYRight)
        xRight +=50
        hatWidth = int(hypot((xRight - xLeft), (yRight - yLeft)))
        hatScale = hatWidth / w

        overlayImage = cv2.resize(overlayImage, (0, 0), fx=hatScale, fy=hatScale)
        hatHeight, hatWidth, _ = overlayImage.shape  # Size of foreground
        print(self.controlPoints, yLeft)
        if yLeft - hatHeight <= 0 or yLeft >= rows or xLeft <= 0 or xLeft + hatWidth >= cols:
            return img
        bg = img[yLeft - hatHeight: yLeft, xLeft:xLeft + hatWidth]
        np.multiply(bg, np.atleast_3d(255 - overlayImage[:, :, 3]) / 255.0, out=bg, casting="unsafe")

        img[yLeft - hatHeight: yLeft, xLeft:xLeft + hatWidth] = bg

        return img


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    listImg = os.listdir(ROOT)
    print(listImg)
    imgList = []
    for imgPath in listImg:
        img = cv2.imread(f'{ROOT}\\{imgPath}', cv2.IMREAD_UNCHANGED)
        imgList.append(img)
    print(len(imgList))
    indexImg = 0

    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        # Set draw=True if you would like to see face mesh
        img = detector.findFaceMesh(img, draw=False)

        if (len(imgList)):
            detector.transparentOverlay(img, imgList[indexImg])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('a'):
            if indexImg > 0:
                indexImg -= 1
        elif key == ord('d'):
            if indexImg < len(imgList) - 1:
                indexImg += 1
        elif key == ord('q'):
            break


if __name__ == "__main__":
    main()