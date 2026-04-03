import os
import pickle
from src.embeddings.clip_model import get_clip_embedding

REFERENCE_PATH = "reference"

def build_reference_embeddings():
    reference_embeddings = {}

    for category in os.listdir(REFERENCE_PATH):
        category_path = os.path.join(REFERENCE_PATH, category)

        # skip if not folder
        if not os.path.isdir(category_path):
            continue

        embeddings = []

        for img in os.listdir(category_path):

            # 🔥 IMPORTANT FIX — only images allow
            if not img.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(category_path, img)

            emb = get_clip_embedding(img_path)
            embeddings.append(emb)

        reference_embeddings[category] = embeddings

    # save embeddings
    with open("reference_embeddings.pkl", "wb") as f:
        pickle.dump(reference_embeddings, f)

    print("✅ Reference embeddings created successfully!")

build_reference_embeddings()