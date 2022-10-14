
from .cfggeneral import GeneralConfig

class CNNConfig(GeneralConfig):

    def __init__(self):
        super(CNNConfig, self).__init__()
        
        # args: [img_size, lr, milestones, batchsize, base_save_weights, epoch, weight_decay, gamma, train_layer_idx(after this)]
        self.ferckynet_args = [192, 1e-3, [20, 40, 60], 64, '_ferckynet.pth', 70, 0.001, 0.1, 0]
        self.xalexnet_args = [224, 3e-4, [27, 59], 32, '_alexnet.pth', 70, 0.001, 0.1, 8]
        self.xdensenet121_args = [224, 3e-4, [20, 25], 32, '_densenet121.pth', 30, 0.001, 0.1, 336]
        self.xgooglenet_args = [224, 3e-4, [20, 25], 32, '_googlenet.pth', 30, 0.0001, 0.1, 135]
        self.xmobilenetv3_args = [224, 3e-4, [20, 25], 32, '_mobilenetv3.pth', 30, 0.0001, 0.1, 141]
        self.xresnet18_args = [224, 3e-4, [20, 25], 32, '_resnet18.pth', 30, 0.001, 0.1, 45]
        self.xresnet50_args = [224, 3e-4, [20, 25], 32, '_resnet50.pth', 30, 0.001, 0.1, 129]
        self.xvgg11_args = [224, 3e-4, [20, 25], 32, '_vgg11.pth', 30, 0.001, 0.1, 12]
        self.xvgg16_args = [224, 3e-4, [20, 25], 32, '_vgg16.pth', 30, 0.001, 0.1, 18]

        # self.ferckynet_args = [192, 1e-3, [20, 40, 60], 64, '_ferckynet_sr.pth', 70, 0.0001, 0.1, 0]
        # # self.ferckynet_args = [192, 1e-2, [2, 7, 13], 64, '_ferckynet_sr_step1.pth', 15, 0.0001, 0.1, 0]
        # self.xalexnet_args = [224, 3e-4, [27, 59], 32, '_alexnet_sr.pth', 70, 0.001, 0.1, 8]
        # self.xdensenet121_args = [224, 3e-4, [20, 25], 32, '_densenet121_sr.pth', 30, 0.001, 0.1, 336]
        # self.xgooglenet_args = [224, 3e-4, [20, 25], 32, '_googlenet_sr.pth', 30, 0.0001, 0.1, 135]
        # self.xmobilenetv3_args = [224, 3e-4, [20, 25], 32, '_mobilenetv3_sr.pth', 30, 0.0001, 0.1, 141]
        # self.xresnet18_args = [224, 3e-4, [20, 25], 32, '_resnet18_sr.pth', 30, 0.001, 0.1, 45]
        # self.xresnet50_args = [224, 3e-4, [20, 25], 32, '_resnet50_sr.pth', 30, 0.001, 0.1, 129]
        # self.xvgg11_args = [224, 3e-4, [20, 25], 32, '_vgg11_sr.pth', 30, 0.001, 0.1, 12]
        # self.xvgg16_args = [224, 3e-4, [20, 25], 32, '_vgg16_sr.pth', 30, 0.001, 0.1, 18]