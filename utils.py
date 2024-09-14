import os.path
from pathlib import Path
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
    plt.savefig(f"{output_dir}/learning_curves.png")
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
    plt.savefig(f"{output_dir}/ROC_curves.png")
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
def process_output_text(path):
    output_path = os.path.join(Path(path).parent, os.path.basename(path)[:-4]+"_processed.txt")
    lines = open(path,"r", encoding='utf-8').readlines()
    new_lines = [l for l in lines if "|" not in l]
    with open(output_path,'w+') as out:
        for l in new_lines:
            out.write(l)
        out.close()

output_text = r"C:\Users\soldier109\Documents\Technion\Semester 10\094295 - Lab in Data Visualization\Project\code\output\130924_1913.txt"
process_output_text(output_text)