import os
import torch
from data_loader import load_data
from model import ActiveLearningVgg16
from active_learning import active_learning_loop
from visualization import create_summary_table, visualize_results
from Config import Config


def run_experiment():
    """
    Conducts experiments with different models and sampling methods.
    Returns:
        dict: Results of the experiments.
    """

    results = {}
    config = Config()
    torch.manual_seed(config.seed)
    for model_name in config.MODELS:
        config.update_model_name(model_name)
        results[model_name] = {}
        model = ActiveLearningVgg16()

        for method in config.SAMPLING_METHODS:
            train_loader_labeled, train_loader_unlabeled, val_loader, test_loader = load_data(
                config.DATA_DIR,
                batch_size=config.BATCH_SIZE,
                labeled_unlabeled_split=config.TRAIN_LABELED_UNLABELED_RATIO,
                total_train_samples=config.TOTAL_TRAINING_SAMPLES,
                total_test_samples=config.TOTAL_TEST_SAMPLES,
                seed=config.seed)

            CURR_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, f"{model_name}_{method}")
            print(f"Running experiment with {model_name} and {method} sampling...")

            # Initialize results storage for this vgg16 and method
            results[model_name][method] = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}

            # Initialize and run Active Learning
            metrics = active_learning_loop(
                model, train_loader_labeled, val_loader, test_loader, unlabeled_data=train_loader_unlabeled,
                method=method,
                config=config,
                output_dir=CURR_OUTPUT_DIR)

            # Evaluate vgg16 after each iteration and store results
            for met in metrics:
                for key in results[model_name][method]:
                    results[model_name][method][key].append(met[key])

    # Visualize results
    summary_df = create_summary_table(results, output_dir=config.OUTPUT_DIR)
    summary_table = visualize_results(summary_df, output_dir=config.OUTPUT_DIR)
    config.write_config_to_file()
    return summary_table
