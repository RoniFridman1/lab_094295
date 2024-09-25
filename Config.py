class Config:
    def __init__(self,model_name):
        self.model_name = model_name

        if model_name == 'resnet18':
            self.leaning_rate = 1e-5
        if model_name == 'vgg16':
            self.leaning_rate = 1e-6