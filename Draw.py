# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2

import numpy as np
import matplotlib.pyplot as plt

w = 640
h = 480
fig = plt.figure(figsize=(15, 10))

columns = 4
rows = 4

image_name = '000050.jpg'
augment_names = ['horizental_flip', 'vertical_flip', 'scale', 'hue', 'saturation', 'gray', 'brightness', 'gaussian', 'shift', 'crop']

ax = []
for i in range(columns*rows):
    image_path = './res/{}_{}.jpg'.format(image_name, augment_names[i])
    image = cv2.imread(image_path)
    image = cv2.resize(image, (w, h), interpolation = cv2.INTER_CUBIC)

    # create subplot and append to ax
    ax.append( fig.add_subplot(rows, columns, i+1) )

    ax[-1].set_xticklabels([])
    ax[-1].set_yticklabels([])
    ax[-1].set_title('# {}'.format(augment_names[i]))

    plt.imshow(image[..., ::-1])

plt.show()