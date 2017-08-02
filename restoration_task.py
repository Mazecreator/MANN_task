from config import SAVE_DIR_RESTORE_TASK, PICTURE_DIR_RESTORE_TASK, MNISTRestoreConfig
from utils import images_print, create_dir
from ops import mnist_loader
from cell import MANNCell
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging

import os

logging.basicConfig(format = "[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MNISTRestore(MNISTRestoreConfig): 
    def __init__(self):
        MNISTRestoreConfig.__init__(self)
        logger.info("Building model starts...")
        self.input = tf.placeholder(tf.float32, shape= [self.batch_size, self.nseq, self.chunk_dim])
        self.target = tf.placeholder(tf.float32, shape= [self.batch_size, self.nseq, self.chunk_dim])
        self.predict = self.build_up_rnn(self.input)
        self.square_error = tf.reduce_mean(tf.square(self.predict-self.target))
        logger.info("Building model done.")
        self.dataload()
        print("Variables in cell\n    "+"\n    ".join(["{} : {}".format(v.name, v.get_shape().as_list()) for v in self.rnn_cell.vars]))
        self.run_train = tf.train.AdamOptimizer().minimize(self.square_error, var_list = self.rnn_cell.vars)
        self.sess = tf.Session()

    def dataload(self):
        logger.info("Dataloading starts")
        self.train_data, self.test_data, self.val_data = mnist_loader()
        logger.info("Dataloading done")

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def restore(self):
        logger.info("Restoring model starts")
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(SAVE_DIR_RESTORE_TASK, 'model'))
        logger.info("Restoring model done")

    def build_up_rnn(self, x):
        x = tf.transpose(x, [1,0,2]) # [nseq, batch_size, chunk_di]
        x = tf.reshape(x, [-1, self.chunk_dim]) # [nseq*batch_size, chunk_dim]
        x = tf.split(x, self.nseq, axis = 0) # nseq x [batch_size, chunk_dim]
        with tf.variable_scope("rnn") as scope:
            self.rnn_cell = MANNCell(mem_size=self.mem_size, mem_dim=self.mem_dim, controller_dim=self.controller_dim, output_dim=self.output_dim ,batch_size=self.batch_size, n_reads=self.n_reads, name='MANN')
            outputs, self.states = tf.contrib.rnn.static_rnn(self.rnn_cell, x, initial_state=self.rnn_cell.initial_state(), dtype=tf.float32) 
        outputs_stack = tf.stack(outputs)
        return tf.transpose(tf.sigmoid(outputs_stack), [1, 0 , 2])

    def train(self):
        logger.info("Training starts")
        saver = tf.train.Saver(max_to_keep=10)
        self.batch_num = len(self.train_data['input'])//self.batch_size

        for e in range(self.epoch):
            index = np.arange(len(self.train_data['input']))
            np.random.shuffle(index)
            trX = self.train_data['input'][index]
            
            epoch_loss = 0 
            for b_ in tqdm(range(self.batch_num), ascii = True, desc = "batch"):
                train_batch = trX[b_*self.batch_size:(b_+1)*self.batch_size]
                train_batch_r = np.reshape(train_batch, [self.batch_size, 28, 28])
                input_batch = np.transpose(np.transpose(train_batch_r, [1, 0, 2])[:-1], [1,0,2])
                output_batch = np.transpose(np.transpose(train_batch_r, [1, 0, 2])[1:], [1,0,2])
                train_feed_dict = {self.input : input_batch, self.target : output_batch}
                _, cost = self.sess.run([self.run_train, self.square_error], feed_dict=train_feed_dict)
                epoch_loss += cost
            logger.info('Epoch({}/{}) cost = {}'.format(e+1, self.epoch, epoch_loss))
            if e%self.save_every == self.save_every-1:
                save_path = os.path.join(SAVE_DIR_RESTORE_TASK, 'model')
                saver.save(self.sess, save_path)
                logger.info("Saved in %s"%save_path)
        logger.info("Training done.")

    def test(self, start_index = 0, keep_num = 14):
        def images_generator(images, keep_num = keep_num):
            image_r = np.reshape(images, [self.batch_size, 28, 28])
            image_restore = np.zeros((self.batch_size, 28, 28))
            for i in range(self.batch_size):
                for j in range(keep_num):
                    for k in range(28):
                        image_restore[i][j][k] = image_r[i][j][k] 

            for j in range(keep_num, 28):
                restored = self.sess.run(self.predict, feed_dict ={self.input : image_restore[:,:-1,:]})
                for i in range(self.batch_size):
                    for k in range(28):
                        image_restore[i][j][k] = restored[i][j-1][k]
            return image_restore

        image_sample = np.reshape(self.test_data['input'][start_index:start_index+self.batch_size], [self.batch_size, 28, 28])
        image_clip = np.zeros((self.batch_size, 28, 28))
        for i in range(self.batch_size):
            for j in range(keep_num):
                for k in range(28):
                    image_clip[i][j][k] = image_sample[i][j][k]
        image_fake = images_generator(image_sample, keep_num)
        images_print(image_sample, os.path.join(PICTURE_DIR_RESTORE_TASK, 'image_sample.png'))
        images_print(image_clip, os.path.join(PICTURE_DIR_RESTORE_TASK, 'image_clip.png'))
        images_print(image_fake, os.path.join(PICTURE_DIR_RESTORE_TASK, 'image_fake.png'))
        logger.info("Images are saved in %s"%PICTURE_DIR_RESTORE_TASK)

if __name__=='__main__':
    create_dir(SAVE_DIR_RESTORE_TASK)
    create_dir(PICTURE_DIR_RESTORE_TASK)
    model = MNISTRestore()
    #model.initialize()
    model.restore()
    #model.train()
    model.test()