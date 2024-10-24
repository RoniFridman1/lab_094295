from model import ActiveLearningVgg16, ActiveLearningResnet18
from data_loader import load_data
from Config import Config
import time
import os
from visualization import create_summary_table, visualize_results

config = Config()
model = ActiveLearningResnet18()
baseline_output_dir = os.path.join(config.OUTPUT_DIR, f"baseline")

train_loader_labeled, _, val_loader, test_loader = load_data(
    data_dir=config.DATA_DIR,
    batch_size=config.BATCH_SIZE,
    labeled_unlabeled_split=config.TRAIN_LABELED_UNLABELED_RATIO,  # should be like (0.999, 0.001)
    total_train_samples=config.TOTAL_TRAINING_SAMPLES,
    total_test_samples=config.TOTAL_TEST_SAMPLES,
    total_val_samples=config.TOTAL_VAL_SAMPLES,
    seed=config.seed)

metrics = []
for j in range(config.ACTIVE_LEARNING_ITERATIONS):
    t0 = time.time()
    print(f"Iteration {j + 1}/{config.ACTIVE_LEARNING_ITERATIONS}.")

    # Train the vgg16
    model = model.train(train_loader_labeled, val_loader, epochs=config.MODEL_TRAINING_EPOCHS, learning_rate=config.learning_rate)

    # Evaluate the vgg16
    metrics.append(model.calculate_metrics(test_loader, iteration=j, output_dir=baseline_output_dir))
    print(f"Time of Iteration: {round(time.time() - t0)} sec")

# Evaluate vgg16 after each iteration and store results
results = {"resnet18": {}}
results["resnet18"]["baseline"] = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}
for met in metrics:
    for key in results["resnet18"]["baseline"]:
        results["resnet18"]["baseline"][key].append(met[key])

# Visualize results
summary_df = create_summary_table(results, output_dir=config.OUTPUT_DIR)
visualize_results(summary_df, output_dir=config.OUTPUT_DIR)
config.write_config_to_file()
