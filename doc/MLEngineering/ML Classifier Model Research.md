**Question: Will our database contain many people? How will this affect which model we pick**

* Large databases could mean increased processing time, which model is optimized for that?   
* Accuracy vs speed trade-off: If we go for a fast model you may sacrifice some recognition accuracy.

**OpenCV and VidGear**  
Details: 

* Vidgear Libraries  
  * FFmpeg  
    * Convert video into processable data for input (Adelin’s team)  
  * 

Resources: 

**Possible Models**

1. Integrating custom model  
   1. Specs?  
2. Haar Cascade Classifier \- face detection  
   1. File: haarcascade\_frontalface\_default.xml  
   2. Found in: opencv/data/haarcascades/  
   3. Algorithm: Haar feature-based cascade classifier.  
   4. Pros: Fast, lightweight.  
   5. Cons: Not as accurate as modern DNNs.  
3. DNN Face Detector (ResNet-based) \- face detection  
   1. Files:  
      1. Deploy.prototxt  
      2. res10\_300x300\_ssd\_iter\_140000.caffemodel  
   2. Model: Single Shot Multibox Detector (SSD) with ResNet-10 backbone.  
   3. Pros: Much more accurate than Haar.  
   4. Cons: Slightly slower.  
   5. [opencv/samples/dnn/face\_detector at master · opencv/opencv](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)  
4. MTCNN \- face detection (finding the faces)  
   1. [ipazc/mtcnn: MTCNN face detection implementation for TensorFlow, as a PIP package.](https://github.com/ipazc/mtcnn)  
   2. Needs GPU, slower on CPU  
   3. Higher accuracy than OpenCV’s DNN SSD  
   4. Handles angles and lighting better than DNN SSD  
5. YOLOv8-Face \- face detection  
   1. Highest accuracy when compared to DNN and MTCNN  
   2. More resource usage though  
   3. Fast  
   4. Typically used for real-time \+ production  
   5. May not be the easiest  
6. DeepFace \- facial recognition  
   1. Wraps multiple backends, but isn’t very customizable  
   2. Requires GPU and CPU  
7. [FaceNet](https://github.com/davidsandberg/facenet)  
   1. Outputs embedding vector  
      1. 128 dimension vector  
      2. Trained to have the embedding vector for the same people to be close (euclidean distance-wise)  
   2. Triplet loss  
   3. Better for simple, lightweight setups  
   4. Simpler than ArcFace  
8. ArcFace  
   1. Outputs embedding vector  
      1. 512 dimension vector  
   2. Uses angular margins instead of triplet loss  
   3. Outperforms FaceNet but typically used with large-scale data  
   4. Not great for limited hardware

**Frameworks (maybe)**

1. PyTorch  
   1. Can increase accuracy  
   2. Will need more setup and memory \- possible slower runtime  
   3. Higher accuracy \- Slower model  
   4. Favored in research \- considered more flexible?  
   5. User friendly  
2. TensorFlow   
   1. Steeper learning curve  
   2. Considered better for scalability and deployment?

**General pipeline:**

1. Detect if there is a face in frame \- facial detection (Custom model)  
   1. Process input (numpy array of faces)  
   2. Determine features  
      1. Which ones to prioritize? Which ones to ignore?   
      2. Train to ignore unnecessary factors (darkness, background objects, etc.)  
         1. We can use attention to determine which features are more important than others, and use it to prioritize certain features  
   3. Test on given images  
      1. From Adelin, we will receive processed images in the form of csv files  
   4. Cross test with pretrained models for accuracy  
      1. Get pretrained model from OpenCV/VidGear  
      2. Compare accuracy  
2. Compare embeddings to known embeddings and determine if it matches  
   1. We can train an SVM or KNN here\!  
   2. Distance based \- euclidean, cosine

**Additional Resources:**  
[Building a Computer Vision Model with OpenCV and Python](https://codezup.com/building-computer-vision-model-opencv-python/)  
[Ganesh-KSV/vit-face-recognition-1 at main](https://huggingface.co/Ganesh-KSV/vit-face-recognition-1/tree/main)  
[https://www.geeksforgeeks.org/computer-vision/feature-extraction-and-image-classification-using-opencv/](https://www.geeksforgeeks.org/computer-vision/feature-extraction-and-image-classification-using-opencv/)  
→[https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images)  
