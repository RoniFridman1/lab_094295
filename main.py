from experiment import run_experiment

if __name__ == "__main__":
    DATA_DIR = "chest_xray"
    MODELS = ["resnet18"]
    # MODELS = ["resnet18","vgg16"]
    SAMPLING_METHODS = ["uncertainty", "entropy",'random']
    ACTIVE_LEARNING_ITERATIONS = 10
    MODEL_TRAINING_EPOCHS = 3
    SAMPLES_PER_ITERATION = 10
    TOTAL_TRAINING_SAMPLES = 250
    TOTAL_TEST_SAMPLES = 250
    BATCH_SIZE=25
    run_experiment(DATA_DIR, MODELS, SAMPLING_METHODS, ACTIVE_LEARNING_ITERATIONS, MODEL_TRAINING_EPOCHS,
                   SAMPLES_PER_ITERATION, TOTAL_TRAINING_SAMPLES,TOTAL_TEST_SAMPLES,BATCH_SIZE)
