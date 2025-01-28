
from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf






def visual_bits(idx, x, y, x_reconstruction_0, x_reconstruction_1):
    plt.figure(figsize=(5, 5))
    j = 0

    for i in idx:
        # reshape images:
        ori_img = x[i][:784] * 255.0
        res0_img = x_reconstruction_0[i].numpy()
        res1_img = x_reconstruction_1[i].numpy()

        res0_img = res0_img[:784] * 255.0
        res1_img = res1_img[:784] * 255.0

        ori_img = ori_img.reshape((28, 28))
        res0_img = res0_img.reshape((28, 28))
        res1_img = res1_img.reshape((28, 28))

        ori_lbl = str(y.to_numpy()[i].argmax())

        j += 1
        plt.subplot(len(idx), 3, j)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(ori_img, cmap='gray')
        # plt.xlabel(ori_lbl)

        j += 1
        plt.subplot(len(idx), 3, j)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(res0_img, cmap='gray')

        j += 1
        plt.subplot(len(idx), 3, j)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(res1_img, cmap='gray')

        np.set_printoptions(precision=2)

    plt.show()

