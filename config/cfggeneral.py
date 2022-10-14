
class GeneralConfig:

    def __init__(self):
        
        # self.data_path = './srdata/'
        self.data_path = './data/'
        self.weight_root = './weights/'
        self.result_root = './result/'
        self.subdir_list = ['train', 'val', 'test']
        self.tensorboard_log = './runs/'
        self.class_num = 7
        self.class_name = [
            'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
        ]