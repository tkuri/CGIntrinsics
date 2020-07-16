import time
import torch
import numpy as np
from options.train_options import TrainOptions
import sys, traceback
import h5py
from data.data_loader import CreateDataLoader
from models.models import create_model
# from data.data_loader import CreateDataLoader_TEST
from data.data_loader import CreateDataLoader_MI
import cv2
import os

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch


# root = "/home/zl548/phoenix24/"
# full_root = root +'/phoenix/S6/zl548/'
full_root = './'
output_dir = "./CGIntrinsics/Multi-Illumination/results/"

model = create_model(opt)


def test_mi(model, list_name):
    # print("============================= Validation ============================")
    model.switch_to_eval()

    # print("============================= Testing EVAL MODE ============================", j)
    test_list_dir = full_root + '/CGIntrinsics/Multi-Illumination/' + list_name
    print(test_list_dir)
    data_loader_MI = CreateDataLoader_MI(full_root, test_list_dir)
    dataset_mi = data_loader_MI.load_data()

    for i, data in enumerate(dataset_mi):
        stacked_img = data['img_1']
        targets = data['target_1']
        input_img, SH = model.test_mi(stacked_img, targets)

        L_img = targets['L'][0]
        L_img_np = L_img.data[:,:,:].cpu().numpy()
        L_img_np = np.transpose(L_img_np, (1, 2, 0))

        path = targets['path'][0]

        print('targets:', targets['path'])
        tar_dir = os.path.dirname(path)
        src_file = os.path.splitext(os.path.basename(path))[0] + '_input.png'
        tar_file = os.path.splitext(os.path.basename(path))[0] + '_SH.png'
        L_file = os.path.splitext(os.path.basename(path))[0] + '_L.png'

        os.makedirs(output_dir+tar_dir, exist_ok=True)
        cv2.imwrite(output_dir+tar_dir+'/'+src_file, input_img*255.0)
        cv2.imwrite(output_dir+tar_dir+'/'+tar_file, SH*255.0)
        cv2.imwrite(output_dir+tar_dir+'/'+L_file, L_img_np*255.0)
        print('Save {}...'.format(tar_file))

print("WE ARE IN TESTING PHASE!!!!")
test_mi(model, 'train_list/')

print("We are done")
