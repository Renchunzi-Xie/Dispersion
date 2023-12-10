from models.resnet import *
from models.wrn import *

def get_model(arch, num_classes, seed):
    if arch == 'resnet18':
        model = ResNet18(num_classes=num_classes, seed=seed)
    elif arch == 'resnet50':
        model = ResNet50(num_classes=num_classes, seed=seed)
    elif arch == 'wrn_50_2':
        model = wrn_50_2(num_classes=num_classes, seed=seed)
    else:
        raise Exception("Not Implemented Error")
    return model
