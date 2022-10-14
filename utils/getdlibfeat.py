
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import cv2
import dlib
from skimage import io

def img2feat(img_path, predictor, convertor):

    img_rd = io.imread(img_path)
    # img_rd = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    img_rd = cv2.cvtColor(img_rd, cv2.COLOR_GRAY2RGB)
    faces = dlib.rectangle(0, 0, img_rd.shape[0], img_rd.shape[1])
    shape = predictor(img_rd, faces)
    face_descriptor = convertor.compute_face_descriptor(img_rd, shape)
    return face_descriptor