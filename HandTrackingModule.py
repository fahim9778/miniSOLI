import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5,
                 trackingConfindence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfindence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils  # for drawing hand landmark dots

    def findHands(self, image, draw=True):
        # converting RGB and sending to hands object
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks) # printing hand landmarks dots

        if self.results.multi_hand_landmarks:
            for handLandMarks in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, handLandMarks,
                                           self.mpHands.HAND_CONNECTIONS)  # putting landmark dots & connecting
                # them on original image
        return image

    def findPosition(self, image, handNumber=0, draw=True):

        landMarkList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]

            for pointId, landmark in enumerate(myHand.landmark):
                # print(pointId, landmark)
                height, width, channel = image.shape
                centerX, centerY = int(landmark.x * width), int(landmark.y * height)
                #print(pointId, centerX, centerY)
                landMarkList.append([pointId, centerX, centerY])
                # if pointId == 4:
                if draw:
                    cv2.circle(image, (centerX, centerY), 5, (255, 255, 255),
                           cv2.FILLED)  # fill the specific hand id point

        return landMarkList




def main():
    # 0 === PC webcam
    # 1 === other cam (e.g: DroidCam client)
    vidCap = cv2.VideoCapture(0)  # capturing video from webcam
    detector = handDetector()

    # for FPS calculation
    prevTime = 0
    currTime = 0

    while True:
        success, img = vidCap.read()  # capturing video from webcam
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # fps calculation
        # currTime = time.time()
        # fps = 1 / (currTime - prevTime)
        # prevTime = currTime
        #
        # cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
