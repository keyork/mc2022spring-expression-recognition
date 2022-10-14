
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import numpy as np
import cv2
from skimage import io
import dlib
'''
dlib_face_recognition_resnet_model_v1.dat
shape_predictor_68_face_landmarks.dat
'''
predictor = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
face_feature_128D = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')
img_rd = io.imread('./demo_img.jpg')
print(img_rd.shape)
img_rd = cv2.cvtColor(img_rd, cv2.COLOR_GRAY2RGB)
print(img_rd.shape)
print(np.array([img_rd, img_rd]).shape)
# img_rd = np.array([img_rd, img_rd])
faces = dlib.rectangle(0, 0, img_rd.shape[0], img_rd.shape[1])
shape = predictor(img_rd, faces)
face_descriptor = face_feature_128D.compute_face_descriptor(img_rd, shape)

print(face_descriptor)
print(type(face_descriptor))
print(torch.tensor(face_descriptor))