import serial
import cv2 as cv



BAUD_RATE = 115200
PORT = "/dev/cu.usbmodem101"

#initialize the connection between arduino sensors and python
ser = serial.Serial("/dev/cu.usbmodem101", BAUD_RATE)
#print("WE ARE HERE!")

#states
camera_on = False
#current serial data coming in
state = None
#receipt to allow me to use the projector
cap = None

#print("2")
#will change to turn on or off depending on time later
while True:
    #print("3")
    #if we have recieved a new message, then set the current state to what was received
    #from the serial monitor
    #-this keeps the video streaming even if no srial messages arrive
    #-this makes the serial event-based, not loop-blocking
    
    if ser.in_waiting > 0:

        #reads bytes from a serial port, stops when see a\n
        state = ser.readline().decode().strip()
        if (state == "PRESENT" and not camera_on):

            print("PRESENT -> opening camera.")
            #grab the projector from the store
            cap = cv.VideoCapture(0)
            #get the code, this means we're on

            if cap.isOpened():
                camera_on = True
                print("Success: Camera turned on!")


            else:
                cap = None
                print("Error: Camera failed to turned on!")
        
        elif state == "ABSENT" and camera_on:
            print("ABSENT -> turing off camera.")
            #stop video capture
            cap.release()
            #closed all camera windows opened on pc
            cv.destroyAllWindows()
            camera_on = False
            cap = None

    if camera_on and cap is not None:
    #read method returns two parameters: 
    #1. ret-> returns a boolean of whether or not the frame was successfully captured or not
    #2. frame -> typically a numpy representation of the captured frame
        ret, frame = cap.read()
        if(ret):
            #show the capturing image. in this case the frame data from cap.read()
            cv.imshow("Camera", frame)
            #time between each frame capture. measured in ms
            cv.waitKey(1)
        
    
        




