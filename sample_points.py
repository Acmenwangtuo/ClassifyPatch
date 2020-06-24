import os
import sys
import logging
import argparse

import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__), "/../../"))


parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")
parser.add_argument("mask_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the mask npy file")
parser.add_argument("txt_path", default=None, metavar="TXT_PATH", type=str,
                    help="Path to the txt file")
parser.add_argument("--patch_number", default=1000, metavar="PATCH_NUMB", type=int,
                    help="The number of patches extracted from WSI")
parser.add_argument("--level", default=6, metavar="LEVEL", type=int,
                    help="Bool format, whether or not")
parser.add_argument("--patch_size", default=32, metavar="PATCH_SIZE", type=int,
                    help="The size of one tile")

class patch_point_in_mask_gen(object):
    '''
    extract centre point from mask
    inputs: mask path, centre point number
    outputs: centre point
    '''

    def __init__(self, mask_path,number,patch_size):
        self.mask_path = mask_path
        self.number = number
        self.patch_size = patch_size

    def get_patch_point(self):
        mask_tissue = np.load(self.mask_path)
        X_idcs, Y_idcs = np.where(mask_tissue)
        points = [[x,y] for x,y in zip(X_idcs,Y_idcs)]
        sampled_points = []
        patch_size = self.patch_size
        def Judge(x,y):
            '''
            判断以x，y为中心的点是否是一个合格的tile
            '''
            left_top = [int(x - patch_size / 2),int(y - patch_size / 2)]
            # right_top = [int(x - patch_size / 2), int(y + patch_size / 2)]
            # left_bottom = [int(x + patch_size / 2),int( y - patch_size / 2)]
            # right_bottom = [int(x + patch_size / 2), int(y + patch_size / 2)]
            count = np.sum(mask_tissue[left_top[0]:left_top[0]+patch_size,left_top[1]:left_top[1]+patch_size])
            if (count / patch_size ** 2 >= 0.9):
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
                    cnt += 1
                    if cnt > self.number:
                        print(cnt)
                        break
        sampled_points = np.array(sampled_points)      
        # print(len(sampled_points))
     
        # centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)
        # print(centre_points.shape[0])

        # if centre_points.shape[0] > self.number:
        #     sampled_points = centre_points[np.random.randint(centre_points.shape[0],
        #                                                      size=self.number), :]
        # else:
        #     sampled_points = centre_points
        print(sampled_points)
        return sampled_points


def run(args):
    sampled_points = patch_point_in_mask_gen(args.mask_path,args.patch_number,args.patch_size).get_patch_point()
    mama = sampled_points
    sampled_points = (sampled_points * 2 ** (args.level-1)).astype(np.int32) # make sure the factor
    mama = np.array(mama)
    mask_name = os.path.split(args.mask_path)[-1].split(".")[0]
    name = np.full((sampled_points.shape[0], 1), mask_name)
    center_points = np.hstack((name, sampled_points))
    print(center_points[0])
    txt_path = args.txt_path

    with open(txt_path, "a") as f:
        np.savetxt(f, center_points, fmt="%s", delimiter=",")



def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()