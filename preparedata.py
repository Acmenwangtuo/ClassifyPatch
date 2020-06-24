import sys
import os
import argparse
import logging
import time
from shutil import copyfile
from multiprocessing import Pool,Value,Lock

import openslide

parser = argparse.ArgumentParser(description='Generate pathes from a give '
                                'list of coordinates')

parser.add_argument('wsi_path')