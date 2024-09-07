from experiment import run_experiment

if __name__ == "__main__":
    DATA_DIR = "chest_xray"
    MODELS = ["resnet18", "vgg16"]
    SAMPLING_METHODS = ["uncertainty", "entropy"]

    run_experiment(DATA_DIR, MODELS, SAMPLING_METHODS)
