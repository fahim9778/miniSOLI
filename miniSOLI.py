import cv2
import time
import numpy as np
import math
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##############################################
camWidth, camHeight = 1280, 720
##############################################
# 0 === PC webcam
# 1 === other cam (e.g: DroidCam client)

vidCap = cv2.VideoCapture(0) # capturing video from webcam
vidCap.set(3, camWidth)
vidCap.set(4, camHeight)

detector = htm.handDetector(detectionConfidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
minVolume = volumeRange[0]
maxVolume = volumeRange[1]
vol = 0
volBar = 400
volPercent = 0

while True:
    success, img = vidCap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    if len(landmarkList) != 0:
        # print(landmarkList[4], landmarkList[8])

        thumbTipX, thumbTipY = landmarkList[4][1], landmarkList[4][2]
        indexTipX, indexTipY = landmarkList[8][1], landmarkList[8][2]
        centerX, centerY = (thumbTipX + indexTipX) // 2, (thumbTipY + indexTipY) // 2

        cv2.circle(img, (thumbTipX, thumbTipY), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (indexTipX, indexTipY), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (thumbTipX, thumbTipY), (indexTipX, indexTipY), (255, 0, 255), 3)
        cv2.circle(img, (centerX, centerY), 10, (255, 255, 255), cv2.FILLED)

        lineLength = math.hypot((indexTipX - thumbTipX), (indexTipY - thumbTipY))
        # print(lineLength)

        # Hand Range 50 ~ 300
        # Volume Range -65 ~ 0

        vol = np.interp(lineLength, [50, 250], [minVolume, maxVolume])
        volBar = np.interp(lineLength, [50, 300], [400, 150])
        volPercent = np.interp(lineLength, [50, 300], [0, 100])
        print(int(lineLength), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if lineLength < 50:
            cv2.circle(img, (centerX, centerY), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPercent)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Video Scene", img)
    cv2.waitKey(1)
