# author: Sami Ibrahim
# local prototype for facial detection system
# pipeline: capture ---> pre-processing ---> detection ---> display 

import cv2

#Open default webcam hence the "0"
#Instatiating the object capture of the VideoCapture Class
capture = cv2.VideoCapture(0)

#Checking if the camera was opened successfully 
if not capture.isOpened():
    print("FAILED: attempted to open default webcamera on device")
    exit()

#Load Haar cascade which is a pre-trained face detector
#For the project the ML group is in charge of developing and training the classifier but for now I'm using
#the pre-trained model "haarcascades" as a placeholder
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#self-explanatory just naming the window
window_title = "Boring Club Face Detection"
cv2.namedWindow(window_title)

while True:
    #Capture an image frame by frame as the webcam runs in the form of a 3D matrix (array) of pixels
    success, frame = capture.read()

    #Check if a frame was successfully captured
    if not success:
        print("ERROR: failed to grab frame")
        break

    # PRE-PROCESSING
    # Convert frame to grayscale for detection 
    #We convert to shades of black -> 0 and white -> 255 to simplify computation
    #i.e. you can imagine BGR (Blue Green Red) colors would have way too many differing values
    #the return is a 2D array of pixels 
    #Fun fact (for linear algebra): We perform a linear transformation collapsing the 3D matrix into a single intensity value in a 2D matrix
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #DETECTION
    #--------------------------------------------------------------------------------------
    #Detect faces using the haarcascades
    #This method scans the image at different sizes based on the scale factor provided
    #returns a corresponding (positioning, width, height) of the face
    #--------------------------------------------------------------------------------------
    #EXPLANATION (optional):
    #Because different face sizes exist (relative to distance from camera) rescaling is necessary
    #Agreements ("this is *a face") at different scales are recorded and if there is overlap at different
    #(minNeighbors) e.g. 5 scales then their positioning (position, width, height ) is returned
    #1.1 means factor of 10% per rescale, 5 means minimum of 5 overlaps
    #greater scalefactor means less accurate but faster, slower more accurate but slower
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    THICKNESS = 1
    COLOR = (0, 255, 0)
    #Draw rectangles around faces
    #rectangle(frame, position1, position2, color, thickness)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR, THICKNESS)

    #DISPLAY
    #-------------------------------
    #Display the frame on the window
    cv2.imshow(window_title, frame)

    #Exit on 'q' key or if window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
        break

#Release the camera and close all windows to free resources
capture.release()
cv2.destroyAllWindows()