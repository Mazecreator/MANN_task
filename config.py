#=========PATH========#
SAVE_DIR_RESTORE_TASK = './save/restore'
PICTURE_DIR_RESTORE_TASK = './asset/restore'

#========Class configurations=====#
class MNISTRestoreConfig(object):
    def __init__(self):
        # Hyper parameter
        self.batch_size = 100
        self.epoch = 10
        self.mem_size = 28
        self.mem_dim = 7
        self.n_reads = 2
        self.save_every = 5
        self.controller_dim= 128
        self.output_dim = 28
        self.chunk_dim = 28
        self.nseq = 27