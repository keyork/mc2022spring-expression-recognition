
def load_config(config, network_name):

    if network_name == 'ferckynet':
        train_args = config.ferckynet_args
    elif network_name == 'alexnet':
        train_args = config.xalexnet_args
    elif network_name == 'densenet121':
        train_args = config.xdensenet121_args
    elif network_name == 'googlenet':
        train_args = config.xgooglenet_args
    elif network_name == 'mobilenetv3':
        train_args = config.xmobilenetv3_args
    elif network_name == 'resnet18':
        train_args = config.xresnet18_args
    elif network_name == 'resnet50':
        train_args = config.xresnet50_args
    elif network_name == 'vgg11':
        train_args = config.xvgg11_args
    elif network_name == 'vgg16':
        train_args = config.xvgg16_args
    
    return train_args