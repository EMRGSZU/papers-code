import os

#记录一些log、data的地址
class Setting:

    def __init__(self, ratio):

        self.ratio = ratio

        self.lr = 1e-4
        self.epoch = 100
        self.step = [101]
        self.batch = 8

        self.train_dataset_name = '/data/majunnan/machi/training_set//train2.mat'
        self.val_dataset_name = '/data/majunnan/machi/test_set//BSDS500/data/images/val/'

        self.save_dir = '/data/majunnan/machi/algorithm/'
        self.work_name = '/'.join(os.getcwd().split('/')[5:])
        self.result_dir = self.save_dir + self.work_name + '/{}'.format(self.ratio)
        self.model_dir = self.result_dir + '/model'
        self.pic_dir = self.result_dir + '/pic'
        self.analysis = self.result_dir + '/analysis'
        self.log_file = self.result_dir + 'log_{}.txt'.format(self.ratio)

        self.mkdirs()

    def mkdirs(self):

        if os.path.exists(self.model_dir) == False:
            os.makedirs(self.model_dir)
        if os.path.exists(self.pic_dir) == False:
            os.makedirs(self.pic_dir)
        if os.path.exists(self.analysis) == False:
            os.makedirs(self.analysis)