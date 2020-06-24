import json
import os
import argparse
import logging
import sys

import numpy as np 
from skimage.measure import points_in_poly

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Formatter(object):

    _injson = ''
    _outjson = ''
    _regions = {}

    def __init__(self,injson,outjson):
        self._injson = injson
        self._outjson = outjson

    def  get_cor(self,scale = None):
        with open(self._injson) as f:
            file = json.load(f)
        print(len(file))
        if scale is None:
           scale = 1
        regions = {}
        for reigon in file:
            k = 1
            points = reigon['points']
            coor = {}
            X = []
            Y = []
            for i in range(0,len(points)-1,2):
               X.append(int(points[i]/scale))
               Y.append(int(points[i+1]/scale))
            assert len(X) == len(Y)
            coor['X'] = X
            coor['Y'] = Y
            regions['region'+str(k)] = coor
            k += 1
        self._regions = regions

    def json_tranform(self,scale=None):
        self.get_cor(scale)
        wsi_dict = {
            'Name': os.path.basename(self._injson).split('.')[0],
            'Region_num': (len(self._regions)),
            'Region': self._regions
        }
        json_str = json.dumps(wsi_dict,indent=4)
        with open(self._outjson,'w') as json_file:
            json_file.write(json_str)

    def in_Polygon(self,x,y,scale=None):
        self.get_cor(scale)
        # print(self._regions)
        for k,reigon in self._regions.items():
            X = reigon['X']
            Y = reigon['Y']
            polygon = [[x,y] for x,y in zip(X,Y)]
            vertices = np.array(polygon)
            #print(vertices.shape)
            coord = (x,y)
            if points_in_poly([coord], vertices)[0]:
                continue
            else:
                return False
        return True

            
if __name__ == "__main__":
    json_obj = Formatter('/home/bnc/tool/HistomicsML/yourproject/keras-yolo3/wsi_json/51270003.json','./51270003.json')
    json_obj.json_tranform()
    json_obj.in_Polygon(0,0)


