import sys
import argparse
import base64
import os
import csv
import itertools

csv.field_size_limit(sys.maxsize)

import h5py
import torch.utils.data
import numpy as np
from tqdm import tqdm

import config
import data
import utils


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test', action='store_true')# 处理训练和验证集的特征
    # 处理test集的特征
    parser.add_argument("--test", action='store_false', help='set this to test.')
    args = parser.parse_args()

    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

    features_shape = (      # (123287, 2048, 36)
        82783 + 40504 if not args.test else 81434,
        # number of images in trainval or in test
        config.output_features, # 2048
        config.output_size,     # 36
    )
    
    boxes_shape = (         # (123287, 4, 36)
        features_shape[0],      # 82783 + 40504 if not args.test else 81434
        4,
        config.output_size,     # 36
    )

    features_similarities_shape = ( # (123287, 36, 36)
        features_shape[0],      # 82783 + 40504 if not args.test else 81434
        config.output_size,     # 36
        config.output_size,     # 36
    )

    if not args.test:
        path = config.preprocessed_trainval_path
        # print(path)
        # '/home/gaoling106023/Projects/VQA2.0-Recent-Approachs-2018/data/genome-trainval.h5'  
    else:
        path = config.preprocessed_test_path
        # print(path)
        # '/home/gaoling106023/Projects/VQA2.0-Recent-Approachs-2018/data/genome-test.h5'
    # with h5py.File(path, libver='latest') as fd:
    with h5py.File(path, 'w') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        features_similarities = fd.create_dataset('features_similarities', shape=features_similarities_shape, dtype='float32')
        coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
        widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
        heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')

        readers = []
        if not args.test:
            path = config.bottom_up_trainval_path
            # '/home/gaoling106023/Projects/VQA_data/VQA_V2/image_features/trainval_36'
        else:
            path = config.bottom_up_test_path
            # '/home/gaoling106023/Projects/VQA_data/VQA_V2/image_features/test2015'
        for filename in os.listdir(path):
            if not '.tsv' in filename:
                continue
            full_filename = os.path.join(path, filename)
            fd = open(full_filename, 'r')
            reader = csv.DictReader(fd, delimiter='\t', fieldnames=FIELDNAMES)
            readers.append(reader)

        reader = itertools.chain.from_iterable(readers)
        for i, item in enumerate(tqdm(reader, total=features_shape[0])):
            coco_ids[i] = int(item['image_id'])
            widths[i] = int(item['image_w'])
            heights[i] = int(item['image_h'])

            buf = base64.decodebytes(item['features'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, config.output_features)).transpose()
            features[i, :, :array.shape[1]] = array
            

            buf = base64.decodebytes(item['boxes'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, 4)).transpose()
            boxes[i, :, :array.shape[1]] = array


if __name__ == '__main__':
    main()
