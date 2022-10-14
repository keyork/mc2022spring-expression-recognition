# source activate MC2022
# # python train.py --method 'dlib_feat' --args_idx 0
# # python train.py --method 'dlib_feat' --args_idx 1
# # python train.py --method 'dlib_feat' --args_idx 2
# # # nohup bash run.sh >> train.log 2>&1 &
# # python train.py --method cnn --network ferckynet --pretrained False --debug False
# python train.py --method cnn --network alexnet --pretrained True --debug False
# # python train.py --method cnn --network densenet121 --pretrained True --debug False
# # python train.py --method cnn --network googlenet --pretrained True --debug False
# python train.py --method cnn --network mobilenetv3 --pretrained True --debug False
# # python train.py --method cnn --network resnet18 --pretrained True --debug False
# python train.py --method cnn --network resnet50 --pretrained True --debug False
# # python train.py --method cnn --network vgg11 --pretrained True --debug False
# python train.py --method cnn --network vgg16 --pretrained True --debug False

# # python train.py --method cnn --network ferckynet --pretrained False --debug False --step2 True --pre_model Acc_60_51_ferckynet_final_step1.pth
# # python train.py --method cnn --network vgg16 --pretrained False --debug False --step2 True --pre_model Acc_61_49_vgg16_sr.pth
# # python train.py --method cnn --network ferckynet --pretrained False --debug False --step2 True --pre_model Acc_61_57_ferckynet_final_step3.pth