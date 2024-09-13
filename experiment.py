from data_loader import load_data
from model import initialize_model, train_model
from active_learning import active_learning_loop, evaluate_model
from model_downloader import download_model
import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curves(metrics_history, model_name, sampling_method,output_dir):
    """
    Plots learning curves for a given model and sampling method.

    Args:
        metrics_history (dict): Dictionary containing lists of metrics over iterations.
        model_name (str): Name of the model.
        sampling_method (str): Sampling method used.
    """
    iterations = range(1, len(metrics_history['accuracy']) + 1)

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, metrics_history['accuracy'], label='Accuracy', marker='o')
    plt.plot(iterations, metrics_history['f1_score'], label='F1 Score', marker='o')
    plt.title(f'Learning Curves for {model_name} using {sampling_method}')
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.imsave(f"{output_dir}/learning_curves.png")
    plt.close()


def plot_roc_curves(roc_auc_scores, model_name, sampling_method,output_dir):
    """
    Plots ROC curves for different models or sampling methods.

    Args:
        roc_auc_scores (dict): Dictionary containing ROC-AUC scores over iterations.
        model_name (str): Name of the model.
        sampling_method (str): Sampling method used.
    """
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(roc_auc_scores) + 1)
    plt.plot(iterations, roc_auc_scores, label=f'ROC-AUC for {model_name}', marker='o')

    plt.title(f'ROC-AUC Curve for {model_name} using {sampling_method}')
    plt.xlabel('Iteration')
    plt.ylabel('ROC-AUC Score')
    plt.legend()
    plt.grid(True)
    plt.imsave(f"{output_dir}/learning_curves.png")
    plt.close()


def create_summary_table(results):
    """
    Creates a summary table comparing models and sampling methods.

    Args:
        results (dict): Dictionary containing results for all models and sampling methods.

    Returns:
        pd.DataFrame: Summary table as a DataFrame.
    """
    data = []
    for model_name, sampling_results in results.items():
        for sampling_method, metrics in sampling_results.items():
            row = {
                'Model': model_name,
                'Sampling Method': sampling_method,
                'Final Accuracy': metrics['accuracy'][-1],
                'Final Precision': metrics['precision'][-1],
                'Final Recall': metrics['recall'][-1],
                'Final F1-Score': metrics['f1_score'][-1],
                'Final ROC-AUC': metrics['roc_auc'][-1],
            }
            data.append(row)

    df = pd.DataFrame(data)
    return df


def visualize_results(results, output_dir):
    """
    Visualizes the results using various plots and tables.

    Args:
        results (dict): Dictionary containing results for all models and sampling methods.
    """
    for model_name, sampling_results in results.items():
        for sampling_method, metrics in sampling_results.items():
            plot_learning_curves(metrics, model_name, sampling_method,output_dir)
            plot_roc_curves(metrics['roc_auc'], model_name, sampling_method,output_dir)

    summary_table = create_summary_table(results)
    return summary_table


def run_experiment(data_dir, models, sampling_methods,
                   active_learning_iterations=5, training_epochs=5, samples_per_iteration=32,
                   total_train_samples=128, batch_size = 32):
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
    train_loader_labeled, train_loader_unlabeled, val_loader, test_loader = load_data(
        data_dir, batch_size=batch_size, labeled_unlabeled_split=(0.25, 0.75), total_train_samples=total_train_samples)

    results = {}

    for model_name in models:
        OUTPUT_DIR = f"output/{model_name}_{sampling_methods}"
        results[model_name] = {}
        model = initialize_model(model_name)

        for method in sampling_methods:
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
