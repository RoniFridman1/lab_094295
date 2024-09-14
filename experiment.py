from data_loader import load_data
from model import initialize_model, train_model
from active_learning import active_learning_loop, evaluate_model
from model_downloader import download_model
from utils import visualize_results





def run_experiment(data_dir, models, sampling_methods,
                   active_learning_iterations=5, training_epochs=5, samples_per_iteration=32,
                   total_train_samples=128, total_test_samples=100, batch_size = 32):
    """
    Conducts experiments with different models and sampling methods.

    Args:
        data_dir (str): Directory for the dataset.
        models (list): List of model names (e.g., ['resnet18', 'vgg16']).
        sampling_methods (list): List of sampling methods (e.g., ['uncertainty', 'entropy']).
        iterations (int): Number of Active Learning iterations.
        samples_per_iteration (int): Number of samples to query per iteration.

    Returns:
        dict: Results of the experiments.
    """

    results = {}

    for model_name in models:
        results[model_name] = {}
        model = initialize_model(model_name)

        for method in sampling_methods:
            train_loader_labeled, train_loader_unlabeled, val_loader, test_loader = load_data(
                data_dir, batch_size=batch_size, labeled_unlabeled_split=(0.25, 0.75),
                total_train_samples=total_train_samples, total_test_samples=total_test_samples)

            OUTPUT_DIR = f"output/{model_name}_{method}"
            print(f"Running experiment with {model_name} and {method} sampling...")

            # Initialize results storage for this model and method
            results[model_name][method] = {
                'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []
            }

            # Initialize and run Active Learning
            active_learning_model = active_learning_loop(
                model, train_loader_labeled, val_loader, test_loader, unlabeled_data=train_loader_unlabeled,
                method=method,
                iterations=active_learning_iterations, samples_per_iteration=samples_per_iteration,
                model_train_epochs=training_epochs,
                output_dir = OUTPUT_DIR
            )

            # Evaluate model after each iteration and store results
            for i in range(active_learning_iterations):
                metrics = evaluate_model(active_learning_model, test_loader)
                for key in results[model_name][method]:
                    results[model_name][method][key].append(metrics[key])

    # Visualize results
    summary_table = visualize_results(results,output_dir="output")
    return summary_table
