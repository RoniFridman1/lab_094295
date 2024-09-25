from datetime import datetime
import os

class Config:
    def __init__(self):
        self.seed = 43
        self.model_name = None
        self.leaning_rate = None
        self.DATA_DIR = "chest_xray"
        self.SAMPLING_METHODS = ["uncertainty", "entropy", 'random']
        self.ACTIVE_LEARNING_ITERATIONS = 5
        self.MODEL_TRAINING_EPOCHS = 3
        self.SAMPLES_PER_ITERATION = 25
        self.TOTAL_TRAINING_SAMPLES = 1000
        self.TRAIN_LABELED_UNLABELED_RATIO = (0.1,0.9)
        self.TOTAL_TEST_SAMPLES = 250
        self.BATCH_SIZE = 25

        ## Numbering experiments output folders
        os.makedirs("outputs", exist_ok=True)
        expr_folders_list = sorted([int(x.split("_")[-1]) for x in os.listdir("outputs") if "experiment" in x])
        expr_idx = 1 if len(expr_folders_list) == 0 else expr_folders_list[-1] + 1
        self.OUTPUT_DIR = f"outputs/experiment_{expr_idx}"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def update_model_name(self,model_name):
        self.model_name = model_name

        if model_name == 'resnet18':
            self.leaning_rate = 1e-5
        if model_name == 'vgg16':
            self.leaning_rate = 1e-6

    def config_to_str(self):
        conf_str = f"{self.seed=}\n{self.SAMPLING_METHODS=}\n{self.ACTIVE_LEARNING_ITERATIONS=}\n" +\
            f"{self.MODEL_TRAINING_EPOCHS=}\n{self.SAMPLES_PER_ITERATION=}\n{self.TOTAL_TRAINING_SAMPLES=}\n" +\
            f"{self.TRAIN_LABELED_UNLABELED_RATIO=}\n{self.TOTAL_TEST_SAMPLES}\n{self.BATCH_SIZE=}"
        return conf_str