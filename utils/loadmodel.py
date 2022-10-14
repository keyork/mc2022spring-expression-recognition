
from models.ferckynet import XFerCKYNet
from models.xalexnet import XAlexNet
from models.xdensenet121 import XDenseNet121
from models.xgooglenet import XGoogLeNet
from models.xmobinetv3 import XMobileNetV3
from models.xresnet18 import XResNet18
from models.xresnet50 import XResNet50
from models.xvgg11 import XVGG11
from models.xvgg16 import XVGG16

def load_model(network_name, pretrain, class_num, layer_idx):

    if network_name == 'ferckynet':
        model = XFerCKYNet(class_num)
    elif network_name == 'alexnet':
        model = XAlexNet(class_num, pretrain, layer_idx)
    elif network_name == 'densenet121':
        model = XDenseNet121(class_num, pretrain, layer_idx)
    elif network_name == 'googlenet':
        model = XGoogLeNet(class_num, pretrain, layer_idx)
    elif network_name == 'mobilenetv3':
        model = XMobileNetV3(class_num, pretrain, layer_idx)
    elif network_name == 'resnet18':
        model = XResNet18(class_num, pretrain, layer_idx)
    elif network_name == 'resnet50':
        model = XResNet50(class_num, pretrain, layer_idx)
    elif network_name == 'vgg11':
        model = XVGG11(class_num, pretrain, layer_idx)
    elif network_name == 'vgg16':
        model = XVGG16(class_num, pretrain, layer_idx)
    
    return model