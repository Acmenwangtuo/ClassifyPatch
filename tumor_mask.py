import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json

parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the JSON file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')


def run(args):
    
    # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
    slide = openslide.OpenSlide(args.wsi_path)
    w , h = slide.level_dimensions[args.level]
    mask_tumor = np.zeros((w, h)) # the init mask, and all the value is 0

    # get the factor of level * e.g. level 6 is 2^6
    factor = slide.level_downsamples[args.level]


    with open(args.json_path) as f:
        dicts = json.load(f)
    regions = dicts['Region']

    for _ ,region in regions.items():
        X = region['X']
        Y = region['Y']
        vertices = [[x/factor,y/factor]  for x,y in zip(Y,X)]
        vertices = np.array(vertices)
        vertices = vertices.astype(np.int32)

        cv2.fillPoly(mask_tumor, [vertices], (255))

    mask_tumor = mask_tumor[:] > 127
    mask_tumor = np.transpose(mask_tumor)
    print(mask_tumor.shape)
    np.save(args.npy_path, mask_tumor)

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()