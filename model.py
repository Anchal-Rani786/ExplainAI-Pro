import torchvision.models as models
import torch.nn as nn

def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # disable inplace relu for explainability
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    model.eval()
    return model