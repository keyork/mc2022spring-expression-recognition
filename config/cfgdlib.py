
from .cfggeneral import GeneralConfig

class DlibConfig(GeneralConfig):

    def __init__(self):
        super(DlibConfig, self).__init__()
        self.predictor = 'shape_predictor_68_face_landmarks.dat'
        self.face_descriptor = 'dlib_face_recognition_resnet_model_v1.dat'
        self.expr_list = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.feat_path = 'feature'
        self.train_data = 'train_sr.pth'
        self.val_data = 'val_sr.pth'
        self.test_data = 'test_sr.pth'
        self.model_basename = '_dlib_sr_model.pth'
        # self.train_data = 'train.pth'
        # self.val_data = 'val.pth'
        # self.test_data = 'test.pth'
        # self.model_basename = '_dlib_model.pth'
        self.batch_size = 32
        # args: [lr, milestones, epoch]
        self.args1 = [3e-3, [50, 70, 90], 100]
        self.args2 = [3e-3, [90], 100]
        self.args3 = [3e-3, [70, 140, 170], 200]