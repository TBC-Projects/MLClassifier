# author: Sami Ibrahim
# local prototype for facial recognition system
import cv2
import face_recognition

#.png image format since their lossless compression preserves facial details
image = cv2.imread("C:\Github\BoringClub-Projects\MLClassifier\src\scripts\images\Messi.png")

if image is None:
    raise FileNotFoundError("Could not laod image")

#Converting it from the weird BGR format OpenCV uses to RGB for facial_recognition lib compatability
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#converting the pixel data from the image into its digital format (1 and 0s)
img_encoding = face_recognition.face_encodings(rgb_image)[0]


cv2.imshow("Window", image)
cv2.waitKey(0)