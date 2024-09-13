from experiment import run_experiment

if __name__ == "__main__":
    DATA_DIR = "chest_xray"
    MODELS = ["vgg16","resnet18"]
    SAMPLING_METHODS = ["uncertainty", "entropy",'random']
    ACTIVE_LEARNING_ITERATIONS = 5
    MODEL_TRAINING_EPOCHS = 5
    SAMPLES_PER_ITERATION = 32
    TOTAL_TRAINING_SAMPLES=320
    run_experiment(DATA_DIR, MODELS, SAMPLING_METHODS, ACTIVE_LEARNING_ITERATIONS, MODEL_TRAINING_EPOCHS,
                   SAMPLES_PER_ITERATION, TOTAL_TRAINING_SAMPLES)
