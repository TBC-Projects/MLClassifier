# import cv2
# import os

# #camera is USB 2.0
# camera = cv2.VideoCapture(0) #gets video input from USB port

# #get frame dimensions
# frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# #Defines VideoWriter
# fourcc = cv2.VideoWriter_fourcc(*'h264')
# output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
# image_num = 0

# #video save folder
# #SCRIPT_DIR = os.path.dirname(os.path.abspath(file))
# SCRIPT_DIR = os.getcwd()
# SAVE_FOLDER = os.path.join(SCRIPT_DIR, "saved_images")

# #Create folder if it doesn't exist
# os.makedirs(SAVE_FOLDER, exist_ok=True)

# while True:
#     ret, frame = camera.read() #read first frame, confirm capturing

#     output.write(frame)
#     cv2.imshow('Camera', frame) #shows each frame
#     saved_loc = os.path.join(SAVE_FOLDER, f"frame{image_num}.png")
#     cv2.imwrite(saved_loc, frame)
#     image_num += 1

#     if cv2.waitKey(1)==ord('q'): # 'q' to exit loop
#         break

# #release objects
# camera.release()
# output.release()
# cv2.destroyAllWindows()

import cv2
import time

camera= cv2.VideoCapture(0)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
#try to modify this version with the function in the above code, this is the one on the website, the previous one is in dc
First_Frame = None

while True:
    Check, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if First_Frame is None:
        First_Frame = gray
        continue
    delta_frame = cv2.absdiff(First_Frame, gray)
    Threshold_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]
    Threshold_frame = cv2.dilate(Threshold_frame, None, iterations=2)
    (cntr, _) = cv2.findContours(
        Threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in cntr:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("Frame", frame)
    Key = cv2.waitKey(1)
    if Key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()