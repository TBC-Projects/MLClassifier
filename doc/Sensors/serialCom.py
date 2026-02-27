#serial library, allows dataflow from microcontrollers(ie arduino) to vs code IDE
import serial 
#openCV library, allows manipulation of image/frame capturing
import cv2 as CV

#establish pipeline from microcontroller to main computer
#/dev/cu.usbmodem1401 is the port pathway. micro -> computer
ser = serial.Serial("/dev/cu.usbmodem101")

#want to continuously run the program. always looking for data
while True:
    #returns bytes from the /dev/cu.usbmodem1401 port.
    raw = ser.readline()
    #convert bytes to string of chars
    text = raw.decode()
    #cut out all white space (leading and trailing), leaving only the string of characters
    line = text.strip()

    #camera state
    camera_on = False
    
    #setup the camera
    if(line == "PRESENT" and not camera_on):
        #turn the camera on, prepare for capturing. note: VideoCapture method returns a 
        #VideoCapture object
        print("PRESENT")
        cap = CV.VideoCapture(1)
        #if the camera opened successfully, set the camera_on state TRUE
        if(cap.isOpened()):
            camera_on = True
        else:
            cap = None
            print("Camera failed to open")


    #if object is not in frame AND camera is on, turn it off and clear all
    elif(line == "ABSENT" and camera_on):
        print("ABSENT")
        #turn the camera off
        cap.release()
        #set camera on state to false
        camera_on = False
        #clear all old frames
        CV.destroyAllWindows()
        print("Camera off!")


    if(camera_on):
        #read method returns two parameters: 
        #1. ret-> returns a boolean of whether or not the frame was successfully captured or not
        #2. frame -> typically a numpy representation of the captured frame
        ret,frame = cap.read()
        if(ret):
            #show the capturing image. in this case the frame data from cap.read()
            CV.imshow("CURRENT_WINDOW_DISPLAY_NAME", frame)
            #time between each frame capture. measured in ms
            CV.waitKey(1)
        
