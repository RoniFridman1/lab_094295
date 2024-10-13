import copy
import time
from Config import Config
from torch.utils.data import DataLoader
from select_samples_methods import select_samples


def active_learning_loop(model, train_generator, val_generator, test_generator,
                         unlabeled_data, method, config, output_dir='output'):
    """
    Main loop for Active Learning.

    Args:
        model (ActiveLearningModel): The machine learning vgg16 to be trained.
        train_generator (DataLoader): DataLoader for training data.
        val_generator (DataLoader): DataLoader for validation data.
        test_generator (DataLoader): DataLoader for test data.
        unlabeled_data (DataLoader): DataLoader for the unlabeled data pool.
        method (str): Strategy for sample selection (e.g. 'uncertainty', 'entropy', 'random').
        config (Config.py): Experiment configuration.
        output_dir (str): a directory path where to store the evaluation metrics.

    Returns:
        vgg16 (torch.nn.Module): The trained vgg16 after Active Learning.
    """
    metrics = []
    for j in range(config.ACTIVE_LEARNING_ITERATIONS):
        t0 = time.time()
        train_samples = int(config.TOTAL_TRAINING_SAMPLES * config.TRAIN_LABELED_UNLABELED_RATIO[0] + j * config.SAMPLES_PER_ITERATION)
        unlabeled_samples = int(config.TOTAL_TRAINING_SAMPLES - train_samples)
        print(
            f"Active Learning Iteration {j + 1}/{config.ACTIVE_LEARNING_ITERATIONS}."
            f"\tTrain Samples: {train_samples}"
            f"\tUnlabeled: {unlabeled_samples}")
        iter_model = copy.deepcopy(model)  # We want to start the vgg16 from scratch for every iteration.
        if len(unlabeled_data) <= 0:
            break
        # Train the vgg16 on current labeled data
        iter_model = iter_model.train(train_generator, val_generator, epochs=config.MODEL_TRAINING_EPOCHS,
                                      learning_rate=config.leaning_rate)

        # Select new samples to be labeled
        selected_samples, selected_labels = select_samples(iter_model.vgg16, unlabeled_data, config=config,
                                                           strategy=method, num_samples=config.SAMPLES_PER_ITERATION)

        # Retrieve selected images and labels from the unlabeled dataloader
        train_generator.dataset.indices = train_generator.dataset.indices + [unlabeled_data.dataset.indices[i]
                                                                             for i in
                                                                             range(len(unlabeled_data.dataset.indices))
                                                                             if i in selected_samples]
        updated_train_data = train_generator.dataset

        # Add new labeled samples to the existing training data
        unlabeled_data.dataset.indices = [unlabeled_data.dataset.indices[i]
                                          for i in range(len(unlabeled_data.dataset.indices))
                                          if i not in selected_samples]
        updated_unlabeled_data = unlabeled_data.dataset

        # Recreate train dataloader with updated data
        train_generator = DataLoader(updated_train_data, batch_size=train_generator.batch_size, shuffle=True)
        unlabeled_data = DataLoader(updated_unlabeled_data, batch_size=unlabeled_data.batch_size, shuffle=False)

        # Evaluate the vgg16
        metrics.append(iter_model.calculate_metrics(test_generator, iteration=j, output_dir=output_dir))
        print(f"Time of Iteration: {round(time.time()-t0)} sec")
    return metrics
