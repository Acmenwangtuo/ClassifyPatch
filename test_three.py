import sys
import os
import argparse
import logging
import json
import time
import glob 
import csv

import torch.backends.cudnn as cudnn
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
from torchvision import models
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from tensorboardX import SummaryWriter
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from data.image_produce import ImageDataset
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=10, type=int, help='number of'
                    ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0,1', type=str, help='comma'
                    ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                    ' and GPU_1, default 0.')

class TestImageFolder(Dataset):
    def __init__(self,data_path, img_size,
                 crop_size=224, normalize=True):
        self.images = []
        # glob.glob(r"/media/chenjun/profile/6_data/CCPD2019/*/*.jpg")
        for filename in sorted(glob.glob(data_path)):
            self.images.append('{}'.format(filename))

        self._img_size = img_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)

    def __getitem__(self, index):
        filename = self.images[index]
        # print('haha',filename)
        img = Image.open(os.path.join(filename))

        #resize
        
        img = img.resize((224,224))

        # color jitter
        img = self._color_jitter(img)
        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)

        # PIL image: H W C
        # torch image: C H W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return img, filename

    def __len__(self):
        return len(self.images)

def chose_model(cnn):
    if cnn['model'] == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif cnn['model'] == 'vggnet':
        model = models.vgg16(pretrained=False)
    else:
        raise Exception("I have not add any models. ")
    return model

def test(args):
    with open(args.cnn_path, 'r') as f:
        cnn = json.load(f)
    print(args.save_path)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(os.path.join(args.save_path, 'cnn.json'), 'w') as f:
        json.dump(cnn, f, indent=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))


    model = chose_model(cnn)
    # print(model)
    model.classifier._modules['6'] = nn.Linear(4096, 3)
    # model.fc = nn.Linear(fc_features, 1) # 
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    checkpoint = torch.load('./threeout/best.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    res = {}
    cudnn.benchmark = True
    dataset_test = data.DataLoader(TestImageFolder(cnn['data_path_test'],
                                 cnn['image_size'],
                                 cnn['crop_size'],
                                 cnn['normalize']),
                                 batch_size = 1,
                                 shuffle = False,
                                 num_workers = 10,
                                 pin_memory = False)

    with torch.no_grad():

        for i, (images,filepath) in enumerate(dataset_test):
            image_var = Variable(images.float().cuda(async=True))
            y_pred = model(image_var)
            # output = torch.squeeze(y_pred)
            # probs = output.sigmoid()
            filename = os.path.basename(filepath[0]).split('.')[0]
            _ , output = torch.max(y_pred,1)
            # logging.info('{} vaue is {}'.format(filename,probs))
            output = output.item()
            res[filename] = output
            logging.info("The label of {} is {}".format(filename,output))
            # print(type(res[filename]))
            # print(res[filename])
    logging.info("Writing Predictions to CSV...")
    with open(args.save_path + '/res' + '.csv', 'w') as csvfile:
        fieldnames = ['id', 'label']
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('id', 'label'))
        for row in sorted(res.items()):
            csv_w.writerow(row)
    logging.info("Done")


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()