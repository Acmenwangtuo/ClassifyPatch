import numpy as np
import cv2

import openslide



slide = openslide.OpenSlide('/home/bnc/tool/HistomicsML/yourproject/WSI/2018-51270003.ndpi')
print(slide.level_dimensions)
# img_RGB = np.transpose(np.array(slide.read_region((0, 0),
#                            6,
#                            slide.level_dimensions[6]).convert('RGB')),
#                            axes=[1, 0, 2])

img_RGB = slide.read_region((0,0),6,slide.level_dimensions[6])
img = np.array(img_RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('./tu.png',img)


cv2.circle(img, (625,873), 60,(0, 0, 255), 0)
mask = np.load('./tumor.npy')
tissure = np.load('./tissue.npy')
print('tissue',tissure.shape)
normal = np.load('./normal.npy')
print('normal',normal.shape)
# cv2.imwrite('./yuan.png',img)
normal= normal.astype(int)
normal[normal == True] = 255
cv2.imwrite('./normal.png',normal)
print(mask.shape)
print(img.shape)
mask = mask.astype(int)
mask[mask == True] = 255
cv2.imwrite('./tumor.png',mask)
print(mask)