import matplotlib.pyplot as plt
import numpy as np
import os

def images_print(images, save_name=None):
    '''
    Args :
        images - 3D array [batch, 28, 28]
            batch should be square
    '''
    batch = len(images)
    row = int(np.sqrt(batch))
    if row*row == batch:
        images_r = np.reshape(images, [row,row, 28,28])
        imap = np.transpose(images_r, [0,2,1,3])
        imap_r = np.reshape(imap, [28*row, 28*row])
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(imap_r, cmap='gray')
        plt.axis('off')
        if save_name is None:
            plt.show()        
        else:
            plt.savefig(save_name)
            plt.close(fig)
        return

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)