"""
Train k-NN classifier on MediaPipe landmark features from member folders.
Saves model and label index <-> member_id mapping.
"""
import argparse
import sys
from pathlib import Path

import json
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from feature_extraction import FaceFeatureExtractor
from member_db import get_member_folders


def collect_dataset(members_path=None, model_path="face_landmarker.task", base_dir=None):
    """
    For each member folder, load images, extract features, collect (X, y).
    Returns X (n_samples, n_features), y (label indices), label_to_member_id list.
    """
    extractor = FaceFeatureExtractor(model_path=model_path)
    folders = get_member_folders(members_path)
    if base_dir:
        base_dir = Path(base_dir)
        folders = [(mid, name, base_dir / Path(f).name) for mid, name, f in folders]

    label_to_member_id = []
    X_list = []
    y_list = []

    for member_id, name, folder_path in folders:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"Warning: folder not found {folder_path}, skipping member {member_id}")
            continue
        label_idx = len(label_to_member_id)
        label_to_member_id.append(member_id)

        images = sorted(folder_path.glob("*.jpg")) + sorted(folder_path.glob("*.jpeg")) + sorted(folder_path.glob("*.png"))
        for img_path in images:
            try:
                feat = extractor.get_landmark_features_from_path(str(img_path))
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(label_idx)
            except Exception as e:
                print(f"Skip {img_path}: {e}")

    if not X_list:
        raise RuntimeError("No valid face features collected. Check member folders and images.")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, label_to_member_id


def _tune_threshold(clf, scaler, X_scaled, y, label_to_member_id, out_dir):
    """Hold out last 2 samples per class, try threshold grid, print best."""
    from collections import Counter
    n_classes = len(label_to_member_id)
    val_indices = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) >= 2:
            val_indices.extend(idx[-2:].tolist())
        elif len(idx) == 1:
            val_indices.append(idx[0])
    if not val_indices:
        print("Not enough samples for threshold tuning (need at least 1 per class).")
        return
    X_val = X_scaled[val_indices]
    y_val = y[val_indices]
    k = clf.n_neighbors
    best_acc = -1.0
    best_thresh = 15.0
    thresholds = [6, 8, 10, 12, 15, 18, 20, 25, 30]
    print("\nThreshold tuning (validation = last 2 samples per member):")
    for thresh in thresholds:
        correct = 0
        for i in range(len(y_val)):
            dists, indices = clf.kneighbors(X_val[i].reshape(1, -1), n_neighbors=k)
            neighbor_labels = y[indices[0]]
            majority = Counter(neighbor_labels).most_common(1)[0][0]
            mean_dist = float(dists[0].mean())
            if mean_dist <= thresh and majority == y_val[i]:
                correct += 1
        acc = correct / len(y_val)
        print(f"  threshold={thresh}: accuracy={acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    print(f"Suggested --distance-threshold: {best_thresh} (accuracy {best_acc:.2%})")


def main():
    parser = argparse.ArgumentParser(description="Train k-NN face recognizer from member folders")
    parser.add_argument("--members", type=str, default=None, help="Path to members.json")
    parser.add_argument("--model", type=str, default="face_landmarker.task", help="Path to face_landmarker.task")
    parser.add_argument("--out-dir", type=str, default="models", help="Directory to save model and mapping")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for k-NN")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory for member folders (default: same as members.json)")
    parser.add_argument("--tune-threshold", action="store_true", help="After training, run threshold tuning and print suggested distance threshold")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else None
    X, y, label_to_member_id = collect_dataset(
        members_path=args.members,
        model_path=args.model,
        base_dir=args.base_dir,
    )
    print(f"Collected {X.shape[0]} samples, {len(label_to_member_id)} members. Feature dim: {X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = KNeighborsClassifier(n_neighbors=min(args.k, X.shape[0]), weights="distance", metric="euclidean")
    clf.fit(X_scaled, y)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "knn_face_model.joblib"
    mapping_path = out_dir / "label_to_member_id.json"
    scaler_path = out_dir / "scaler.joblib"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(label_to_member_id, f, indent=2)
    print(f"Saved model to {model_path}, scaler to {scaler_path}, mapping to {mapping_path}")

    if args.tune_threshold:
        _tune_threshold(clf, scaler, X_scaled, y, label_to_member_id, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
