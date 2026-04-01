import os
from embeddings.clip_model import get_clip_embedding
from embeddings.resnet_model import get_resnet_embedding
from similarity.cosine import top_k_similarity

DATASET_PATH = "../dataset_flat"

def load_images(folder):
    path = os.path.join(DATASET_PATH, folder)
    return [os.path.join(path, img) for img in os.listdir(path)]

good_images = load_images("good")
bad_images = load_images("bad")

# reference = first 15 good images
reference_images = good_images[:15]

def create_reference_embeddings(model_func):
    return [model_func(img) for img in reference_images]

def evaluate(model_func):
    refs = create_reference_embeddings(model_func)

    good_scores = []
    for img in good_images[15:]:
        emb = model_func(img)
        score = top_k_similarity(emb, refs)
        good_scores.append(score)

    bad_scores = []
    for img in bad_images:
        emb = model_func(img)
        score = top_k_similarity(emb, refs)
        bad_scores.append(score)

    return good_scores, bad_scores