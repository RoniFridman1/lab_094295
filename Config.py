from datetime import datetime
import os


class Config:
    def __init__(self):
        self.seed = 42
        self.start_time = datetime.now()
        self.model_name = None
        self.leaning_rate = 1e-6
        # self.MODELS = ["vgg16"]
        self.MODELS = ["resnet18"]
        self.DATA_DIR = "chest_xray"
        self.SAMPLING_METHODS = ["pca_then_kmeans","random"]
        self.ACTIVE_LEARNING_ITERATIONS = 5
        self.MODEL_TRAINING_EPOCHS = 3
        self.SAMPLES_PER_ITERATION = 25
        self.TOTAL_TRAINING_SAMPLES = 1000  # Max is 5216
        self.TRAIN_LABELED_UNLABELED_RATIO = (0.1, 0.9)
        self.TOTAL_TEST_SAMPLES = 624  # Max is 624
        self.TOTAL_VAL_SAMPLES = 100 # Min is 16
        self.BATCH_SIZE = 25
        self.PCA_N_COMPONENTS = 100  # Original features are 512 for resnet18 and 4096 for vgg16.
        self.K_CLUSTERS = 50

        n_unlabeled_samples_last_iteration = self.TOTAL_TRAINING_SAMPLES * self.TRAIN_LABELED_UNLABELED_RATIO[1] - \
                                             (self.ACTIVE_LEARNING_ITERATIONS - 1) * self.SAMPLES_PER_ITERATION


        ## Checking configuration is valid for any method.
        err1 = f"Not enough unlabeled samples in the last selection iteration (n={n_unlabeled_samples_last_iteration})" + \
               f"to choose from (sample per iteration={self.SAMPLES_PER_ITERATION})"
        err2 = f"Error in total number of training samples. Maximum available is 5216. You chose {self.TOTAL_TRAINING_SAMPLES}."
        err3 = f"Error in total number of test samples. Maximum available is 624. You chose {self.TOTAL_TEST_SAMPLES}."
        assert (self.SAMPLES_PER_ITERATION < n_unlabeled_samples_last_iteration), err1
        assert (self.TOTAL_TRAINING_SAMPLES <= 5216), err2
        assert (self.TOTAL_TEST_SAMPLES <= 624), err3

        ## Checking that configuration is valid for "PCA then Kmeans" method.
        if "pca_then_kmeans" in self.SAMPLING_METHODS:
            prefix = "'PCA then KMeans' sampling method: "
            err1 = prefix + f"Make sure that the number of \nlabeled training " + \
                   "samples is greater or equal to the number of samples per iteration.\n" + \
                   f"Labeled Training Samples: {self.TOTAL_TRAINING_SAMPLES * self.TRAIN_LABELED_UNLABELED_RATIO[0]}" + \
                   f"\tSample per Iteration: {self.SAMPLES_PER_ITERATION}"

            err2 = prefix + f"Not enough unlabeled samples in the last selection iteration.\n" + \
                   f"Will try to cluster {n_unlabeled_samples_last_iteration} samples to K={self.K_CLUSTERS} clusters."
            err3 = prefix + f"Number of cluster should be at least the number of samples per iteration,\n" + \
                   "Since we choose one sample of each the top n_samples_per_iteration clusters."

            assert (self.TOTAL_TRAINING_SAMPLES * self.TRAIN_LABELED_UNLABELED_RATIO[
                0] >= self.SAMPLES_PER_ITERATION), err1
            assert (self.K_CLUSTERS <= n_unlabeled_samples_last_iteration), err2
            assert (self.K_CLUSTERS >= self.SAMPLES_PER_ITERATION), err3

        # Numbering experiments output folders
        os.makedirs("outputs", exist_ok=True)
        expr_folders_list = sorted([int(x.split("_")[-1]) for x in os.listdir("outputs") if "experiment" in x])
        expr_idx = 1 if len(expr_folders_list) == 0 else expr_folders_list[-1] + 1
        self.OUTPUT_DIR = f"outputs/experiment_{expr_idx}"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def update_model_name(self, model_name):
        self.model_name = model_name
        if model_name == 'resnet18':
            self.leaning_rate = 1e-5
        if model_name == 'vgg16':
            self.leaning_rate = 1e-6

    def write_config_to_file(self):
        conf_str = (
            f"Start time={self.start_time}\n"
            f"Seed={self.seed}\n"
            f"Model Name={self.model_name}\n"
            f"Learning rate={self.leaning_rate}\n"
            f"MODELS={self.MODELS}\n"
            f"DATA_DIR={self.DATA_DIR}\n"
            f"SAMPLING_METHODS={self.SAMPLING_METHODS}\n"
            f"ACTIVE_LEARNING_ITERATIONS={self.ACTIVE_LEARNING_ITERATIONS}\n"
            f"MODEL_TRAINING_EPOCHS={self.MODEL_TRAINING_EPOCHS}\n"
            f"SAMPLES_PER_ITERATION={self.SAMPLES_PER_ITERATION}\n"
            f"TOTAL_TRAINING_SAMPLES={self.TOTAL_TRAINING_SAMPLES}\n"
            f"TRAIN_LABELED_UNLABELED_RATIO={self.TRAIN_LABELED_UNLABELED_RATIO}\n"
            f"TOTAL_TEST_SAMPLES={self.TOTAL_TEST_SAMPLES}\n"
            f"TOTAL_VAL_SAMPLES={self.TOTAL_VAL_SAMPLES}\n"
            f"BATCH_SIZE={self.BATCH_SIZE}\n"
            f"PCA_N_COMPONENTS={self.PCA_N_COMPONENTS}\n"
            f"K_CLUSTERS={self.K_CLUSTERS}\n"
            f"OUTPUT_DIR={self.OUTPUT_DIR}\n"
        )

        # Writing the configuration to a file
        with open(os.path.join(self.OUTPUT_DIR, "expr_config.txt"), "w+") as f:
            f.write(conf_str)
            f.close()
