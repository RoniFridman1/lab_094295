from experiment import run_experiment

if __name__ == "__main__":
    DATA_DIR = "chest_xray"
    MODELS = ["vgg16","resnet18"]
    SAMPLING_METHODS = ["uncertainty", "entropy",'random']
    ACTIVE_LEARNING_ITERATIONS = 3
    MODEL_TRAINING_EPOCHS = 2
    SAMPLES_PER_ITERATION = 5
    TOTAL_TRAINING_SAMPLES=40
    BATCH_SIZE=5
    run_experiment(DATA_DIR, MODELS, SAMPLING_METHODS, ACTIVE_LEARNING_ITERATIONS, MODEL_TRAINING_EPOCHS,
                   SAMPLES_PER_ITERATION, TOTAL_TRAINING_SAMPLES,BATCH_SIZE)
# data_dir, models, sampling_methods,
#                    active_learning_iterations=5, training_epochs=5, samples_per_iteration=32,
#                    total_train_samples=128, batch_size = 32