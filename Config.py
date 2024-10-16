from datetime import datetime
import os


class Config:
    def __init__(self):
        self.seed = 42
        self.start_time = datetime.now()
        self.model_name = None
        self.leaning_rate = None
        self.MODELS = ["vgg16"]
        self.DATA_DIR = "chest_xray"
        self.SAMPLING_METHODS = ["random"]
        self.ACTIVE_LEARNING_ITERATIONS = 2
        self.MODEL_TRAINING_EPOCHS = 1
        self.SAMPLES_PER_ITERATION = 10
        self.TOTAL_TRAINING_SAMPLES = 21  # max 5216
        self.TRAIN_LABELED_UNLABELED_RATIO = (0.1, 0.9)
        self.TOTAL_TEST_SAMPLES = 624  # max 624
        self.BATCH_SIZE = 25
        self.PCA_N_COMPONENTS = 3
        # in order for the code to work with core_set you must assert that:
        # TOTAL_TRAINING_SAMPLES * TRAIN_LABELED_UNLABELED_RATIO[0] > SAMPLES_PER_ITERATION

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
        conf_str = f"Start time={self.start_time}\nSeed={self.seed}\n" \
                   f"SAMPLING_METHODS={self.SAMPLING_METHODS}\n" + \
                   f"ACTIVE_LEARNING_ITERATIONS={self.ACTIVE_LEARNING_ITERATIONS}\n" \
                   f"MODEL_TRAINING_EPOCHS={self.MODEL_TRAINING_EPOCHS}\n" + \
                   f"SAMPLES_PER_ITERATION={self.SAMPLES_PER_ITERATION}\n" \
                   f"TOTAL_TRAINING_SAMPLES={self.TOTAL_TRAINING_SAMPLES}\n" + \
                   f"TRAIN_LABELED_UNLABELED_RATIO={self.TRAIN_LABELED_UNLABELED_RATIO}\n" \
                   f"TOTAL_TEST_SAMPLES={self.TOTAL_TEST_SAMPLES}\n" \
                   f"BATCH_SIZE={self.BATCH_SIZE}"
        with open(os.path.join(self.OUTPUT_DIR, "expr_config.txt"), "w+") as f:
            f.write(conf_str)
            f.close()
