"""
test_model.py  –  Diagnostic tool for FaceEmbeddingNet + embeddings.npy.

Run from the project root:
    python test_model.py

Checks:
  1. Inter-person embedding separation (should be LOW similarity).
  2. Intra-person embedding consistency (should be HIGH similarity).
  3. Per-image recognition accuracy against the dataset folder.
  4. Behaviour under the live-recognition thresholds.
"""

import os
import cv2
import torch
import numpy as np
from model import FaceEmbeddingNet
from utils import preprocess_face, cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(SCRIPT_DIR, "face_model.pth")
DB_PATH       = os.path.join(SCRIPT_DIR, "embeddings.npy")
DATASET_PATH  = os.path.join(SCRIPT_DIR, "dataset")
# ──────────────────────────────────────────────────────────────────────────────

THRESHOLD      = 0.68
MIN_MATCH_RATE = 0.45
SCORE_MARGIN   = 0.10


def load_model_and_db():
    device = "cpu"  # CPU is fine for offline testing
    # FIX: embedding_dim=128 must be explicit
    model = FaceEmbeddingNet(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    db = np.load(DB_PATH, allow_pickle=True).item()
    return model, db, device


def get_embedding(model, device, img):
    with torch.inference_mode():
        tensor = preprocess_face(img, augment=False).unsqueeze(0).to(device)
        return model(tensor).cpu().numpy()[0]


def main():
    print("=" * 62)
    print("  MODEL DIAGNOSTIC")
    print("=" * 62)

    model, db, device = load_model_and_db()
    people = list(db.keys())
    print(f"\nPeople in database: {people}\n")

    # ── 1. Inter-person separation ─────────────────────────────────────────────
    print("1. Inter-person embedding separation (lower = better)")
    print("   Target: < 0.70  |  Warning: > 0.80")
    any_warning = False
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            sim = cosine_similarity(db[people[i]]['mean'], db[people[j]]['mean'])
            flag = ""
            if sim > 0.80:
                flag = "⚠️  WARNING"
                any_warning = True
            elif sim > 0.70:
                flag = "⚠️  CAUTION"
            else:
                flag = "✅"
            print(f"   {people[i]}  vs  {people[j]}: {sim:.4f}  {flag}")
    if any_warning:
        print("   → Consider retraining with more diverse images or a larger margin.")

    # ── 2. Intra-person consistency ────────────────────────────────────────────
    print("\n2. Intra-person consistency (higher = better)")
    print("   Target: avg > 0.80  |  Warning: min < 0.50")
    for person in people:
        embs = db[person]['embeddings']
        if len(embs) < 2:
            print(f"   {person}: only 1 embedding — cannot check")
            continue
        sims = [cosine_similarity(embs[i], embs[i + 1])
                for i in range(min(len(embs) - 1, 9))]
        avg_s, min_s = np.mean(sims), np.min(sims)
        flag = "✅" if min_s >= 0.50 and avg_s >= 0.80 else "⚠️"
        print(f"   {person}: avg={avg_s:.4f}  min={min_s:.4f}  {flag}")

    # ── 3. Per-image recognition accuracy ─────────────────────────────────────
    print("\n3. Per-image recognition accuracy (using mean embeddings)")
    if not os.path.isdir(DATASET_PATH):
        print(f"   Dataset not found at {DATASET_PATH} — skipping.")
    else:
        total, correct = 0, 0
        for person in people:
            person_dir = os.path.join(DATASET_PATH, person)
            if not os.path.isdir(person_dir):
                continue
            images = [f for f in os.listdir(person_dir)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            for img_name in images[:5]:   # test up to 5 per person
                img = cv2.imread(os.path.join(person_dir, img_name), 0)
                if img is None:
                    continue
                try:
                    emb = get_embedding(model, device, img)
                except Exception as e:
                    print(f"   Error on {person}/{img_name}: {e}")
                    continue

                scores = {
                    p: cosine_similarity(emb, data['mean'])
                    for p, data in db.items()
                }
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_name, top_score = ranked[0]

                total += 1
                ok = (top_name == person)
                correct += int(ok)
                status = "✅" if ok else "❌"
                print(f"   {status} {person}/{img_name}  → {top_name} ({top_score:.4f})")
                if not ok:
                    for rank, (n, s) in enumerate(ranked[:3], 1):
                        print(f"        {rank}. {n}: {s:.4f}")

        if total > 0:
            acc = correct / total * 100
            grade = ("⚠️  POOR — retrain" if acc < 60
                     else "⚠️  FAIR — consider more data" if acc < 80
                     else "✅ GOOD" if acc < 95
                     else "✅ EXCELLENT")
            print(f"\n   Accuracy: {correct}/{total} = {acc:.1f}%  {grade}")

    # ── 4. Live-threshold simulation ───────────────────────────────────────────
    print(f"\n4. Live-threshold simulation (threshold={THRESHOLD}, "
          f"min_match_rate={MIN_MATCH_RATE}, margin={SCORE_MARGIN})")
    if os.path.isdir(DATASET_PATH):
        for person in people[:2]:
            person_dir = os.path.join(DATASET_PATH, person)
            if not os.path.isdir(person_dir):
                continue
            images = [f for f in os.listdir(person_dir)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not images:
                continue
            img = cv2.imread(os.path.join(person_dir, images[0]), 0)
            if img is None:
                continue
            emb = get_embedding(model, device, img)

            print(f"\n   {person}/{images[0]}:")
            rows = []
            for db_person, data in db.items():
                sims = [cosine_similarity(emb, e) for e in data['embeddings']]
                avg_s    = float(np.mean(sims))
                n_above  = sum(1 for s in sims if s > THRESHOLD)
                rate     = n_above / len(sims)
                rows.append((db_person, avg_s, rate))

            rows.sort(key=lambda r: r[1], reverse=True)
            best_avg   = rows[0][1]
            second_avg = rows[1][1] if len(rows) > 1 else 0.0
            margin     = best_avg - second_avg

            for db_person, avg_s, rate in rows:
                passes = (avg_s >= THRESHOLD and rate >= MIN_MATCH_RATE
                          and (avg_s == best_avg or margin >= SCORE_MARGIN))
                status = "✅ PASS" if passes else "❌ FAIL"
                print(f"      {db_person}: avg={avg_s:.4f}  rate={rate:.2f}  {status}")
            print(f"      Margin between top-2: {margin:.4f}")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()