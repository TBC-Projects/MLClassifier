import cv2

#camera is USB 2.0
camera = cv2.VideoCapture(0) #gets video input from USB port

#get frame dimensions
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Defines VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'h264')
output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = camera.read() #read first frame, confirm capturing

    output.write(frame)
    cv2.imshow('Camera', frame) #shows each frame

    if cv2.waitKey(1)==ord('q'): # 'q' to exit loop
        break

#release objects
camera.release()
output.release()
cv2.destroyAllWindows()