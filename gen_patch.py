
import sys
import os
import argparse
import logging
import time
from shutil import copyfile
from multiprocessing import Pool, Value, Lock
import cv2
import numpy as np
import openslide
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Generate patches from a given '
                                 'list of coordinates')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input directory of WSI files')
parser.add_argument('coords_path', default=None, metavar='COORDS_PATH',
                    type=str, help='Path to the input list of coordinates')
parser.add_argument('patch_path', default=None, metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--patch_size', default=1024, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=1, type=int, help='level for WSI, to '
                    'generate patches, default 0')
parser.add_argument('--num_process', default=5, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(opts):
    i, pid, x_center, y_center, args = opts
    x = int(int(x_center) - args.patch_size / 2)
    y = int(int(y_center) - args.patch_size / 2)
    wsi_path = os.path.join(args.wsi_path, '2018-' + '51270003' + '.ndpi')
    slide = openslide.OpenSlide(wsi_path)

    # img_RGB = slide.read_region((0,0),1,slide.level_dimensions[1])
    # img = np.array(img_RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    img = slide.read_region(
        (y*2,x*2), args.level,
        (args.patch_size, args.patch_size)).convert('RGB')
    # hehe = cv2.imread('./tu.png')
    # cv2.circle(hehe, (y//32,x//32), 32,(0, 0, 255), 0)
    # cv2.imwrite('./res.png',hehe)
    # img = img[x:x+1024,y:y+1024,...]
    img.save(os.path.join(args.patch_path, str(i) + '.png'))
    # cv2.imwrite(os.path.join(args.patch_path, str(i) + '.png'),img)
    global lock
    global count

    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))


def run(args):
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.patch_path):
        os.mkdir(args.patch_path)

    copyfile(args.coords_path, os.path.join(args.patch_path, 'list.txt'))

    opts_list = []
    infile = open(args.coords_path)
    for i, line in enumerate(infile):
        pid, x_center, y_center = line.strip('\n').split(',')
        opts_list.append((i, pid, x_center, y_center, args))
    infile.close()

    pool = Pool(processes=args.num_process)
    pool.map(process, opts_list)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()