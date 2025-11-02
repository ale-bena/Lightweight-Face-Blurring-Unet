import cv2
import numpy as np
import argparse
import csv
from glob import glob
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# app = FaceAnalysis()
app = FaceAnalysis(name="buffalo_l") 
app.prepare(ctx_id=-1, det_size=(128,128))  # CPU, use ctx_id=0 for GPU

def get_aligned_embedding(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img)  # detection + landmarks + embedding
    if len(faces) == 0:
        return None, None
    face = faces[0]
    emb = face.embedding
    bbox = face.bbox.astype(int)
    return emb, bbox

def main(args):
    orig_paths = sorted(glob(args.origin))
    blur_paths = sorted(glob(args.blur))

    similarities = []
    results = []
    print(f"Found {len(orig_paths)} original images, {len(blur_paths)} blurred images")
    # print("Example:", orig_paths[:2], blur_paths[:2])

    for op, bp in zip(orig_paths, blur_paths):
        emb_o, bbox_o = get_aligned_embedding(op)
        emb_b, bbox_b = get_aligned_embedding(bp)

        if emb_o is None or emb_b is None:
            continue

        sim = float(cosine_similarity(emb_o.reshape(1, -1),
                                      emb_b.reshape(1, -1))[0, 0])
        similarities.append(sim)
        results.append({
            "filename_orig": op.split("/")[-1],
            "filename_blur": bp.split("/")[-1],
            "similarity": sim,
        })

    # Save pairwise similarities to CSV
    csv_path = args.csv_path
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename_orig", "filename_blur", "similarity"])
        writer.writeheader()
        writer.writerows(results)

    print("Num pairs:", len(similarities))
    print("Mean cosine similarity:", np.mean(similarities))
    print("Std:", np.std(similarities))
    print(f"CSV saved in {csv_path}")

    threshold = 0.4
    if len(similarities) > 0:
        below_thr = sum(s < threshold for s in similarities) / len(similarities) * 100
    else:
        below_thr = 0.0
    print(f"Percentage of pairs with sim < {threshold}: {below_thr:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Compute cosine similarity between embeddings of original and blurred images using InsightFace."))
    parser.add_argument("--origin", type=str, default="dataset/test/*.jpg", help="Path to original images folder",)
    parser.add_argument("--blur", type=str, default="dataset/test_blur/*.png", help="Path to blurred images folder",)
    parser.add_argument("--csv_path", type=str, default="test_anonym.csv", help="CSV output path",)
    main(parser.parse_args())
