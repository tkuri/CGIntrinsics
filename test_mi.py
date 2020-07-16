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

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch


# root = "/home/zl548/phoenix24/"
# full_root = root +'/phoenix/S6/zl548/'
full_root = './'

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
        SH = model.evlaute_mi(stacked_img, targets)

        print('Save SH{}.png...'.format(i))
        cv2.imwrite(output_dir+'SH{}.png'.format(i), SH*255.0)

print("WE ARE IN TESTING PHASE!!!!")
test_mi(model, 'train_list/')

print("We are done")
