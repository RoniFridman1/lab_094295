<h3 align='center' style="text-align:center; font-weight:bold; font-size:2.5em">Data Analysis and Presentation Lab (094295)</h3>
<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em">Final Project: Enhancing X-Ray Imaging Analysis with Active Learning</h1>

<p align='center' style="text-align:center;font-size:1em;">
    <a>Nimrod Solomon</a>&nbsp;,&nbsp;
    <a>Matan Shiloni</a>&nbsp;,&nbsp;
    <a>Roni Fridman</a>&nbsp;
</p>

### Abstract
X-Rays, one of the most used imaging modalities, play a crucial role in diagnosing various medical conditions. 
However, the effectiveness of machine learning models in this area is often limited by the availability of labeled 
datasets, which are expensive and time-consuming to produce, as they require expert annotation. Our project focuses 
on leveraging Active Learning (AL) for the sake of developing a system that prioritizes the annotation of the most 
informative and challenging X-Ray images, thereby reducing the overall labeling effort required while improving the 
model's performance. Through experiments utilizing various sampling methods, including a novel "PCA then K-Means" 
approach, we found that this method consistently equalized traditional techniques in accuracy, F1-score, and precision.
Our work demonstrates that Active Learning can achieve comparable accuracy to models trained on entire datasets, even
with significantly fewer labeled samples. This project showcases the feasibility of implementing Active Learning in 
pediatric radiology and highlights opportunities for future research.

### Setup Instructions

#### 1. Clone the repository:
```shell
git clone https://github.com/RoniFridman1/lab_094295.git
```

#### 2. Install dependencies:
```shell
pip install copy torch time os datetime torchvision numpy warnings json sklearn matplotlib pandas
```

#### 3. Adjust the Following Config.py parameters to desired values:
```python
        self.seed = 42
        self.learning_rate = 1e-6
        self.MODELS = ["resnet18"]  # options: resnet18, vgg16 (one or more)
        self.SAMPLING_METHODS = ["random"]  # options: pca_then_kmeans, random, uncertainty, entropy (one or more)
        self.ACTIVE_LEARNING_ITERATIONS = 10
        self.MODEL_TRAINING_EPOCHS = 3
        self.SAMPLES_PER_ITERATION = 90
        self.TOTAL_TRAINING_SAMPLES = 5216  # Max is 5216
        self.TRAIN_LABELED_UNLABELED_RATIO = (0.019, 0.981)  # select any between (0.0, 1.0) and (1.0, 0.0)
        self.TOTAL_TEST_SAMPLES = 624  # Max is 624
        self.TOTAL_VAL_SAMPLES = 16  # Min is 16
        self.BATCH_SIZE = 25
        self.PCA_N_COMPONENTS = 100  # Original features are 512 for resnet18 and 4096 for vgg16.
```

#### 4. Run Experiment:
To run an active learning experiment:
```shell
python3 run main.py
```

To run a baseline experiment:
```shell
python3 baseline.py
```




