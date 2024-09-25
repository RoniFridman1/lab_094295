import os

from data_loader import load_data
from model import initialize_model, train_model
from active_learning import active_learning_loop, _evaluate_model
from model_downloader import download_model
from utils import visualize_results
from Config import Config

def run_experiment(models):
    """
    Conducts experiments with different models and sampling methods.

    Args:
        data_dir (str): Directory for the dataset.
        models (list): List of model names (e.g., ['resnet18', 'vgg16']).
        sampling_methods (list): List of sampling methods (e.g., ['uncertainty', 'entropy']).
        active_learning_iterations (int): Number of Active Learning iterations.
        training_epochs (int): number training epochs performed in each iteration.
        samples_per_iteration (int): Number of samples to query per iteration.
        total_train_samples (int): number of train samples.
        total_test_samples (int): number of test samples.
        batch_size (int): batch size argument for the data loader

    Returns:
        dict: Results of the experiments.
    """

    results = {}
    config = Config()
    for model_name in models:
        config.update_model_name(model_name)
        results[model_name] = {}
        model = initialize_model(model_name)

        for method in config.SAMPLING_METHODS:
            train_loader_labeled, train_loader_unlabeled, val_loader, test_loader = load_data(
                config.DATA_DIR, batch_size=config.BATCH_SIZE, labeled_unlabeled_split=config.TRAIN_LABELED_UNLABELED_RATIO,
                total_train_samples=config.TOTAL_TRAINING_SAMPLES, total_test_samples=config.TOTAL_TEST_SAMPLES)

            CURR_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR,f"{model_name}_{method}")
            print(f"Running experiment with {model_name} and {method} sampling...")

            # Initialize results storage for this model and method
            results[model_name][method] = {
                'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }

            # Initialize and run Active Learning
            metrics = active_learning_loop(
                model, train_loader_labeled, val_loader, test_loader, unlabeled_data=train_loader_unlabeled,
                method=method,
                model_name=model_name,
                config=config,
                output_dir=CURR_OUTPUT_DIR)

            # Evaluate model after each iteration and store results
            for met in metrics:
                for key in results[model_name][method]:
                    results[model_name][method][key].append(met[key])

    # Visualize results
    summary_table = visualize_results(results, output_dir=config.OUTPUT_DIR)
    return summary_table
