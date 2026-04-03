import pickle
from src.embeddings.clip_model import get_clip_embedding
from src.similarity.cosine import top_k_similarity

# load embeddings
with open("reference_embeddings.pkl", "rb") as f:
    reference_embeddings = pickle.load(f)

def get_clip_score(image_path, category):
    emb = get_clip_embedding(image_path)

    refs = reference_embeddings[category]

    score = top_k_similarity(emb, refs)

    return score