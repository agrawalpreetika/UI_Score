from run_experiment import evaluate
from embeddings.clip_model import get_clip_embedding
from embeddings.resnet_model import get_resnet_embedding

def analyze(model_name, model_func):
    good, bad = evaluate(model_func)

    good_avg = sum(good) / len(good)
    bad_avg = sum(bad) / len(bad)

    print(f"\n--- {model_name} ---")
    print("Good Avg:", good_avg)
    print("Bad Avg:", bad_avg)
    print("Gap:", good_avg - bad_avg)

analyze("CLIP", get_clip_embedding)
analyze("ResNet", get_resnet_embedding)