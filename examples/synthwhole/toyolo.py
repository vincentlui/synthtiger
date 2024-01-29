import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter
import numpy as np
import synthtiger.utils.image_util as image_util
import os
import shutil
import argparse

GT2LABEL = {
    '1': 9,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    '9': 8,
    '0': 0,
    'q': 16,
    'w': 13,
    'e': 14,
    'r': 15,
    'a': 10,
    's': 12,
    'd': 11,
}
IMAGE_DIM = [640, 640]

def save_files(key, filename, in_img_dir, out_img_dir, out_label_dir, image_dict, gt_dict, bbox_dict):
    shutil.copy2(os.path.join(in_img_dir,image_dict[key]), os.path.join(out_img_dir, filename+'.jpg'))
    label_filename = os.path.join(out_label_dir, filename+'.txt')
    with open(label_filename, 'w') as f:
        gts = gt_dict[key]
        bboxes = bbox_dict[key]
        for gt, bbox in zip(gts, bboxes):
            line = ' '.join(map(str,[gt] + list(bbox))) + '\n'
            f.write(line)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        type=str,
        help="Directory path to save data.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        metavar="DIR",
        type=str,
        help="Directory path of data.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        metavar="DIR",
        type=str,
        help="Prefix of filename.",
    )
    args = parser.parse_args()

    # pprint.pprint(vars(args))

    return args

if __name__ == '__main__':
    args = parse_args()
    gt_dict = {}
    bbox_dict = {}
    image_dict = {}
    image_key_set = {}
    data_dir = args.data_dir
    with open(os.path.join(data_dir,'gt.txt'), 'r') as f:
        with open(os.path.join(data_dir,'coords.txt'), 'r') as g:
            for line in f:
                assert '\n' in line
                split = line[:-1].split('\t')
                image_key = split[0].split('/')[-1].split('.')[0]
                labels = [GT2LABEL[x] for x in split[1:]]
                gt_dict[image_key] = labels
                image_dict[image_key] = split[0]

            for line in g:
                split2 = line[:-1].split('\t')
                image_key2 = split2[0].split('/')[-1].split('.')[0]
                bboxes = [list(map(int, x.split(','))) for x in split2[1:]]
                bboxes = np.array(bboxes)
                center_x = (bboxes[:,2] + bboxes[:,0])/IMAGE_DIM[1]/2
                center_y = (bboxes[:,3] + bboxes[:,1])/IMAGE_DIM[0]/2
                width = (bboxes[:,2] - bboxes[:,0])/IMAGE_DIM[1]
                height = (bboxes[:,3] - bboxes[:,1])/IMAGE_DIM[0]
                bboxes_new = np.vstack([center_x, center_y, width, height]).T
                assert np.all(bboxes_new <= 1) and  np.all(bboxes_new >= 0)
                bbox_dict[image_key2] = bboxes_new

        image_keys = set(gt_dict.keys()).union(set(bbox_dict.keys()))
        assert len(gt_dict.keys()) == len(image_keys)
        assert len(bbox_dict.keys()) == len(image_keys)

        keys = gt_dict.keys()
        num_samples = len(keys)
        num_train = int(.8 * num_samples)
        num_val = int(.2 * num_samples)
        train_keys = list(gt_dict.keys())[:num_train]
        val_keys = list(gt_dict.keys())[num_train: num_train+num_val]
        test_keys = list(gt_dict.keys())[num_train+num_val:]

        output_dir = args.output
        train_image_dir = os.path.join(output_dir,'train/images')
        train_label_dir = os.path.join(output_dir,'train/labels')
        val_image_dir = os.path.join(output_dir,'val/images')
        val_label_dir = os.path.join(output_dir,'val/labels')
        test_image_dir = os.path.join(output_dir,'test/images')
        test_label_dir = os.path.join(output_dir,'test/labels')
        dirs = [train_image_dir, train_label_dir, val_image_dir, val_label_dir, test_image_dir, test_label_dir]

        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
            else:
                print('Directory exists warning')

        prefix = args.prefix
        data_dir = args.data_dir
        for key in train_keys:
            save_files(key, prefix+str(key), data_dir, train_image_dir, train_label_dir, image_dict, gt_dict, bbox_dict)
        for key in val_keys:
            save_files(key, prefix+str(key), data_dir, val_image_dir, val_label_dir, image_dict, gt_dict, bbox_dict)
        for key in test_keys:
            save_files(key, prefix+str(key), data_dir, test_image_dir, test_label_dir, image_dict, gt_dict, bbox_dict)