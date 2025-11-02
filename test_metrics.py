import cv2
import numpy as np
from glob import glob
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import csv

# 1. Inizializza FaceAnalysis (contiene detector + align + ArcFace)
app = FaceAnalysis()
app.prepare(ctx_id=-1, det_size=(128,128))  # CPU
#app = insightface.app.FaceAnalysis(name="antelopev2")  # contiene ArcFace r100
#app.prepare(ctx_id=0, det_size=(128, 128))  # ctx_id=-1 per CPU

def get_aligned_embedding(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img)  # detection + landmarks + embedding
    if len(faces) == 0:
        return None, None
    face = faces[0]
    emb = face.embedding  # già L2-normalizzato
    bbox = face.bbox.astype(int)
    return emb, bbox

# 2. Percorsi immagini originali e blur
orig_paths = sorted(glob("dataset/testset/*.jpg"))
blur_paths = sorted(glob("dataset/unet0210keras/*.png"))

similarities = []
results = []

for op, bp in zip(orig_paths, blur_paths):
    emb_o, bbox_o = get_aligned_embedding(op)
    emb_b, bbox_b = get_aligned_embedding(bp)

    if emb_o is None or emb_b is None:
        continue

    #sim = float(np.dot(emb_o, emb_b))  # cosine similarity
    sim = float(cosine_similarity(emb_o.reshape(1, -1), emb_b.reshape(1, -1))[0,0])
    similarities.append(sim)
    results.append({
        "filename_orig": op.split("/")[-1],
        "filename_blur": bp.split("/")[-1],
        "similarity": sim
    })

with open("Skeras_test_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename_orig", "filename_blur", "similarity"])
    writer.writeheader()
    writer.writerows(results)

print("Num pairs:", len(similarities))
print("Mean cosine similarity:", np.mean(similarities))
print("Std:", np.std(similarities))
print("CSV salvato in csv/unetv0210keras_log.csv")
# Percentuale sotto soglia (es. 0.4)
threshold = 0.4
below_thr = sum(s < threshold for s in similarities) / len(similarities) * 100
print(f"Percentuale di coppie con sim < {threshold}: {below_thr:.2f}%")
