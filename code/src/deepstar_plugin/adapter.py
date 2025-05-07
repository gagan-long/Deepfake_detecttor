from deepstar.sdk import evaluate_model

def benchmark(video_path):
    return evaluate_model(
        video_path=video_path,
        metrics=['accuracy', 'f1_score'],
        dataset_version='v2.1'
    )
