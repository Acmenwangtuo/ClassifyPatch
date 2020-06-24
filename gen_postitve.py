import os
import sys
import logging
import argparse

import numpy as np
import json
sys.path.append(os.path.join(os.path.abspath(__file__), "/../../"))


parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")
# parser.add_argument("mask_path", default=None, metavar="MASK_PATH", type=str,
#                     help="Path to the mask npy file")
parser.add_argument("json_mask_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the mask json file")
parser.add_argument("txt_path", default=None, metavar="TXT_PATH", type=str,
                    help="Path to the txt file")
# parser.add_argument("patch_number", default=None, metavar="PATCH_NUMB", type=int,
#                     help="The number of patches extracted from WSI")
# parser.add_argument("--level", default=6, metavar="LEVEL", type=int,
#                     help="Bool format, whether or not")
parser.add_argument("--patch_size", default=2048, metavar="PATCH_SIZE", type=int,
                    help="The size of one tile")

class patch_point_in_mask_gen(object):
    '''
    extract centre point from mask
    inputs: mask path, centre point number
    outputs: centre point
    '''

    def __init__(self, json_path,patch_size):
        # self.mask_path = mask_path
        # self.number = number
        self.json_path = json_path
        self.patch_size = patch_size

    def get_patch_point(self):
        tumor = 
        points = [[x,y] for x,y in zip(X_idcs,Y_idcs)]
        sampled_points = []
        patch_size = self.patch_size
        def Judge(x,y):
            '''
            判断以x，y为中心的点是否是一个合格的巨响
            '''
            left_top = [x - patch_size / 2, y - patch_size / 2]
            right_top = [x - patch_size / 2, y + patch_size / 2]
            left_bottom = [x + patch_size / 2, y - patch_size / 2]
            right_bottom = [x + patch_size / 2, y + patch_size / 2]
            if left_bottom in points and right_bottom in points and left_top in points and right_top in points :
                return True
            else:
                return False
        x_min,y_min = min(X_idcs) , min(Y_idcs)
        x_max,y_max = max(X_idcs) , max(Y_idcs)
        cnt = 1
        for i in range(x_min,x_max+1,patch_size):
            for j in range(y_min,y_max+1,patch_size):
                if Judge(i,j):
                    sampled_points.append(np.array([i,j]))
                    logging.info('{} has been chosen'.format(cnt))
                    cnt +=1
        sampled_points = np.array(sampled_points)      
        # print(len(sampled_points))
     
        # centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)
        # print(centre_points.shape[0])

        # if centre_points.shape[0] > self.number:
        #     sampled_points = centre_points[np.random.randint(centre_points.shape[0],
        #                                                      size=self.number), :]
        # else:
        #     sampled_points = centre_points
        return sampled_points


def run(args):
    sampled_points = patch_point_in_mask_gen(args.mask_path,args.patch_size).get_patch_point()
    sampled_points = (sampled_points * 2 ** (args.level-1)).astype(np.int32) # make sure the factor

    mask_name = os.path.split(args.mask_path)[-1].split(".")[0]
    name = np.full((sampled_points.shape[0], 1), mask_name)
    center_points = np.hstack((name, sampled_points))

    txt_path = args.txt_path

    with open(txt_path, "a") as f:
        np.savetxt(f, center_points, fmt="%s", delimiter=",")


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()