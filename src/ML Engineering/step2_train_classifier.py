# DO NOT RUN, NOT CONFIRMED TO WORK
# step2_train_classifier.py
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load embeddings
with open("face_embeddings.pkl", "rb") as f:
    embeddings, y, le = pickle.load(f)

# Split for testing
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

# ----- OPTION 1: SVM -----
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
print("SVM Accuracy:", svm.score(X_test, y_test))

# ----- OPTION 2: KNN -----
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
# print("KNN Accuracy:", knn.score(X_test, y_test))

# Save classifier and encoder
with open("face_classifier.pkl", "wb") as f:
    pickle.dump((svm, le), f)

print("✅ Saved trained classifier.")
print(classification_report(y_test, svm.predict(X_test)))
