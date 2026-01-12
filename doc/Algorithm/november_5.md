Collaborating With Teams:
1. Sensors (manage the camera)
2. Actuation (manage the output)
3. Control Hardware (Manage real time data acquisition, input and output)
    -use microcontroller to get the data to us
4. Data science (image processing)
    -handles all pre-processing
5. ML engineering (will create the model that will be used)

What We Need To Do:
1. Stream video data. No pre-processing. 
2. We communicate with data science and Sensors about video quality, resolution (i.e properties of the video)
3. Code how to output is to be shown.

Question to Ask:
1. Control Hardware: What is their planned input/output system
What is the estimated frequency and efficiency of data acquisition?

2. Sensors: What camera are we using? What’s the functionality of the camera? (i.e resolution, processing rate)

3. Actuation: how are we presenting the output (LCD, LED, etc)?

4. ML engineering: What format will the results be?

Process:
Camera gathering the videos --> OpenCV handles live camera feed and prepare image inputs --> ML algorithm deals with images and spits output back to us --> Send the results to hardware teams for physical representation