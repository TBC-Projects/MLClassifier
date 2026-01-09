import cv2
import os

#camera is USB 2.0
camera = cv2.VideoCapture(0) #gets video input from USB port

#get frame dimensions
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Defines VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'h264')
output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
image_num = 0

#video save folder
#SCRIPT_DIR = os.path.dirname(os.path.abspath(file))
SCRIPT_DIR = os.getcwd()
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "saved_images")

#Create folder if it doesn't exist
os.makedirs(SAVE_FOLDER, exist_ok=True)

while True:
    ret, frame = camera.read() #read first frame, confirm capturing

    output.write(frame)
    cv2.imshow('Camera', frame) #shows each frame
    saved_loc = os.path.join(SAVE_FOLDER, f"frame{image_num}.png")
    cv2.imwrite(saved_loc, frame)
    image_num += 1

    if cv2.waitKey(1)==ord('q'): # 'q' to exit loop
        break

#release objects
camera.release()
output.release()
cv2.destroyAllWindows()