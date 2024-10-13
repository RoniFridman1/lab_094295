import os.path
import matplotlib.pyplot as plt
import pandas as pd


def _plot_learning_curves(metrics_history, model_name, sampling_method, output_dir):
    """
    Plots learning curves for a given model and sampling method.

    Args:
        metrics_history (dict): Dictionary containing lists of metrics over iterations.
        model_name (str): Name of the model.
        sampling_method (str): Sampling method used.
        output_dir (str): path for the directory where to save the output plot
    """
    iterations = range(1, len(metrics_history['accuracy']) + 1)

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, metrics_history['accuracy'], label='Accuracy', marker='o')
    plt.plot(iterations, metrics_history['f1_score'], label='F1 Score', marker='o')
    # plt.plot(iterations, metrics_history['roc_auc']*100, label="ROC AUC", marker='o')
    plt.title(f'Learning Curves for {model_name} using {sampling_method}')
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/learning_curves_{model_name}_{sampling_method}.png")
    plt.close()


def _plot_roc_curves(roc_auc_scores, model_name, sampling_method, output_dir):
    """
    Plots ROC curves for different models or sampling methods.

    Args:
        roc_auc_scores (dict): Dictionary containing ROC-AUC scores over iterations.
        model_name (str): Name of the model.
        sampling_method (str): Sampling method used.
        output_dir (str): path for the directory where to save the output plot
    """
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(roc_auc_scores) + 1)
    plt.plot(iterations, roc_auc_scores, label=f'ROC-AUC for {model_name}', marker='o')

    plt.title(f'ROC-AUC Curve for {model_name} using {sampling_method}')
    plt.xlabel('Iteration')
    plt.ylabel('ROC-AUC Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/ROC_curves_{model_name}_{sampling_method}.png")
    plt.close()


def _create_summary_table(results, output_dir):
    """
    Creates a summary table comparing models and sampling methods.

    Args:
        results (dict): Dictionary containing results for all models and sampling methods.
        output_dir (str): path for the directory where to save the summary table as csv file.

    Returns:
        pd.DataFrame: Summary table as a DataFrame.
    """
    data = []
    for model_name, sampling_results in results.items():
        for sampling_method, metrics in sampling_results.items():
            for i in range(len(metrics['accuracy'])):
                row = {
                    'Model': model_name,
                    'Sampling Method': sampling_method,
                    'Active Learning Round': i,
                    'Accuracy': metrics['accuracy'][i],
                    'Precision': metrics['precision'][i],
                    'Recall': metrics['recall'][i],
                    'F1-Score': metrics['f1_score'][i],
                    'ROC-AUC': metrics['roc_auc'][i],
                }
                data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "summary_table.csv"))
    return df


def _plot_summary_table(df, output_path='output/summary'):
    # Ensure the save_location directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Metrics to plot (excluding non-numeric columns)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    # Iterate over each metric to generate one graph per metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # Plot each combination of Model and Sampling Method
        for (model, sampling_method), group in df.groupby(['Model', 'Sampling Method']):
            plt.plot(group['Active Learning Round'], group[metric], label=f'{model} - {sampling_method}')

        # Add labels, title, and legend
        plt.xlabel('Active Learning Round')
        plt.ylabel(metric)
        plt.title(f'{metric} Across Models and Sampling Methods')
        plt.legend(loc='best')

        # Save the plot as a PNG file
        filename = f"{metric}.png"
        filepath = os.path.join(output_path, filename)
        plt.savefig(filepath)

        # Close the plot to avoid overlapping figures in loops
        plt.close()

    print(f"Plots saved at {output_path}")


def visualize_results(results, output_dir):
    """
    Visualizes the results using various plots and tables.

    Args:
        results (dict): Dictionary containing results for all models and sampling methods.
        output_dir: path for the directory where to save the results and visualizations.
    """
    for model_name, sampling_results in results.items():
        for sampling_method, metrics in sampling_results.items():
            _plot_learning_curves(metrics, model_name, sampling_method, output_dir)
            _plot_roc_curves(metrics['roc_auc'], model_name, sampling_method, output_dir)

    summary_table = _create_summary_table(results, output_dir)
    _plot_summary_table(summary_table, os.path.join(output_dir, "summary"))
    return summary_table
