# TBC Pipeline Documentation

The purpose of this document is to figure out some of the ways in which we can transform the data to achieve the facial recognition model from a top down approach. We want to look into how current facial recognition models are built to inspire our own design. 

At the highest level, every approach we are considering is to figure out the most useful representation of a face to send downstream. The differences lie in whether that representation preserves visual appearance, geometry, learned identity semantics, or temporal behavior.

1. Mediapipe  
   1. Mediapipe is used to capture a frame, detect and align the face  
2. Method 1: One of the most common and mature approaches is to transform each detected face into a high-dimensional facial embedding using a pretrained face recognition network.   
   1. The output is a dense numerical vector—often 128, 256, or 512 floating-point values—that encodes the identity-relevant characteristics of that face.   
   2. The embeddings are learned representations where distance in vector space corresponds to identity similarity.   
   3. The machine learning team stores one or more reference embeddings per club member and compares incoming embeddings using cosine similarity or Euclidean distance.   
   4. Attendance is marked once similarity exceeds a confidence threshold over one or more frames.   
   5. This method is extremely popular and performs very well even with moderate lighting or pose variation. It is also relatively privacy-preserving since embeddings cannot be trivially reconstructed into faces.  
3. Method 2: The second method you mentioned is based on passing face images themselves to a convolutional neural network, typically after preprocessing with tools like OpenCV and MediaPipe.   
   1. In this design, the embedded system is responsible only for detecting the face, optionally aligning it using landmarks, and resizing it into a standardized image format.   
   2. That image—often RGB or grayscale—is then sent to the machine learning team, who train and deploy a CNN that learns identity directly from pixel data.   
   3. Unlike embedding-based systems, this approach does not rely on pretrained identity representations; instead, the network learns to distinguish club members as classes or through metric learning.   
   4. This method retains all visual information such as skin texture, facial hair, and accessories, which can improve accuracy in small closed-set scenarios like a club roster. However, it is more sensitive to lighting changes, pose variation, and camera quality, and it requires more data and computation to generalize well.  
4. Method 3: A third approach focuses on geometric facial structure rather than appearance, using MediaPipe’s face mesh or landmark detection capabilities. Here, the embedded system extracts a dense set of facial landmarks—hundreds of points that describe the shape of the face, eyes, nose, lips, and jaw in normalized 2D or 3D space. Instead of sending images, the system transmits these landmark coordinates as numerical data. The machine learning team then treats the face as a structured geometric object rather than a texture-based image. Models trained on this data learn identity from relative distances, proportions, and spatial relationships between facial features. This representation is lightweight, consistent across lighting conditions, and highly privacy-friendly. While it does not capture surface details like freckles or wrinkles, it performs surprisingly well in controlled environments and is particularly well-suited for embedded systems where bandwidth and privacy are constraints.

Mediapipe acts as a high-quality geometric front end that prepares faces so that your embedding model produces consistent, identity-meaningful vectors. Poor alignment and pose variation are among the biggest reasons embedding systems fail and MediaPipe solves that problem.

Step 1: Face detection and tracking using MediaPipe

MediaPipe’s face detection and face mesh models are fast and designed for real-time use on embedded devices. You start by feeding live camera frames into MediaPipe, which returns bounding boxes and facial landmarks for each detected face.

At this stage, MediaPipe gives you:

* A face bounding box

* A dense set of landmarks (typically 468 points)

* Optional tracking IDs across frames

This allows you to detect and track faces efficiently without running expensive neural networks repeatedly.

Step 2: Face alignment using landmarks (critical)

Raw face crops are not sufficient for embeddings. If the face is tilted, rotated, or off-center, embeddings become unstable. MediaPipe landmarks let you fix this.

Using a small subset of landmarks (usually eyes, nose tip, and mouth corners), you compute an affine transform that:

* Rotates the face so the eyes are horizontal

* Scales the face to a fixed size

* Centers key facial features

The result is a **canonical face image**, meaning every face is normalized into the same coordinate space before embedding extraction. This step dramatically improves matching accuracy.

Conceptually, you are using MediaPipe to answer the question: “How should this face be geometrically normalized so identity is preserved?”

Step 3: Cropped and aligned face → embedding model

Once the face is aligned, you pass the resulting image to your face embedding model. This model could be something like MobileFaceNet, ArcFace, or another lightweight architecture suitable for embedded systems.

The output is a fixed-length embedding vector representing the identity of the person. This vector is what you send to the machine learning team or store locally for comparison.

At this point, MediaPipe has done its job: it ensured that the embedding model sees clean, consistent input.

