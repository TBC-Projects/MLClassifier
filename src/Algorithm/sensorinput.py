import cv2
import os
import time


def get_images():
    #camera is USB 2.0
    camera = cv2.VideoCapture(0) #gets video input from USB port

    #get frame dimensions
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Defines VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
    image_num = 0
    last_save = 0

    #video save folder
    SCRIPT_DIR = os.getcwd()
    SAVE_FOLDER = os.path.join(SCRIPT_DIR, "saved_images")

    #Create folder if it doesn't exist
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    while True:
        ret, frame = camera.read() #read first frame, confirm capturing

        output.write(frame)
        cv2.imshow('Camera', frame) #shows each frame

        current_time = time.time()

        if(current_time - last_save >= 1):
            saved_loc = os.path.join(SAVE_FOLDER, f"frame{image_num}.png")
            cv2.imwrite(saved_loc, frame)
            image_num += 1
            last_save = current_time

        if cv2.waitKey(1)==ord('q'): # 'q' to exit loop
            break

    #release objects
    camera.release()
    output.release()
    cv2.destroyAllWindows()

def main():
    get_images()

if __name__ == "__main__":
    get_images()
