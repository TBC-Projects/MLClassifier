from backgroundprocessing import FaceRecognizer, monitor_folder

recognizer = FaceRecognizer(ctx_id=0, threshold=0.6)
recognizer.train("dataset")  # Or use recognizer.load("svm_model.pkl")
recognizer.save("svm_face_model.pkl")
