from experiment import run_experiment

if __name__ == "__main__":
    DATA_DIR = "chest_xray"
    # MODELS = ["resnet18","vgg16"]
    MODELS = ["resnet18",]
    SAMPLING_METHODS = ["uncertainty", "entropy",'random']
    ACTIVE_LEARNING_ITERATIONS = 3
    MODEL_TRAINING_EPOCHS = 1
    SAMPLES_PER_ITERATION = 10
    TOTAL_TRAINING_SAMPLES = 50
    TOTAL_TEST_SAMPLES = 20
    BATCH_SIZE=5
    run_experiment(DATA_DIR, MODELS, SAMPLING_METHODS, ACTIVE_LEARNING_ITERATIONS, MODEL_TRAINING_EPOCHS,
                   SAMPLES_PER_ITERATION, TOTAL_TRAINING_SAMPLES,TOTAL_TEST_SAMPLES,BATCH_SIZE)
