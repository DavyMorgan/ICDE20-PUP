#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags
from absl import logging

import utils
from trainer import Trainer
from tester import InstantTester, PostTester


FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'PUP', 'Experiment name.')
flags.DEFINE_enum('model', 'PUP', ['MF', 'PUP', 'PUPRANK', 'PUP-C', 'PUP-P', 'PUP-CP'], 'Model name.')
flags.DEFINE_enum('mode', 'overall_test', ['train', 'overall_test', 'CIR', 'UCIR', 'ug'], 'Train or test.')
flags.DEFINE_bool('use_gpu', True, 'Use GPU or not.')
flags.DEFINE_integer('gpu_id', 6, 'GPU ID.')
flags.DEFINE_enum('dataset', 'yelp', ['beibei', 'yelp'], 'Dataset.')
flags.DEFINE_integer('num_users', 0, 'Number of users.')
flags.DEFINE_integer('num_items', 0, 'Number of items.')
flags.DEFINE_integer('num_cats', 0, 'Number of categories.')
flags.DEFINE_integer('num_prices', 0, 'Number of price levels.')
flags.DEFINE_string('price_format', 'absolute', 'Price format.')
flags.DEFINE_integer('embedding_size', 64, 'Embedding size for embedding based models.')
flags.DEFINE_float('alpha', 0.5, 'Weight of category branch (1.0 for global branch).')
flags.DEFINE_multi_integer('split_dim', [56, 8], 'Embedding dimension distribution for global branch and category branch.')
flags.DEFINE_integer('epochs', 200, 'Max epochs for training.')
flags.DEFINE_float('lr', 0.01, 'Learning rate.')
flags.DEFINE_float('weight_decay', 5e-8, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout ratio.')
flags.DEFINE_integer('batch_size', 1024, 'Batch Size.')
flags.DEFINE_multi_integer('lr_decay_epochs', [40, 80], 'Epochs at which to perform learning rate decay.')
flags.DEFINE_multi_integer('topk', [50, 100], 'Topk for testing recommendation performance.')
flags.DEFINE_integer('num_workers', 8, 'Number of processes for training and testing.')
flags.DEFINE_string('datafile_prefix', '/data3/zhengyu/price_yelp_restaurant_fm_bpr/', 'Path prefix to data files.')
flags.DEFINE_string('output', '/data3/zhengyu/PUP/', 'Directory to save model/log/metrics.')
flags.DEFINE_integer('port', 33332, 'Port to show visualization results.')
flags.DEFINE_string('workspace', '/data3/zhengyu/PUP/yelp-PUP-test-basePUP-2_2019-07-13-15-58-12/', 'Path to workspace.')


def main(argv):

    flags_obj = flags.FLAGS
    cm = utils.ContextManager(flags_obj)
    vm = utils.VizManager(flags_obj)
    
    if flags_obj.mode == 'train':
    
        cm.set_default_ui()
        vm.show_basic_info(flags_obj)
        trainer = Trainer(flags_obj, cm, vm)
        trainer.train()

        cm.set_test_logging()
        vm.show_test_info(flags_obj)
        tester = InstantTester(flags_obj, trainer.recommender, trainer.cm, vm)
        tester.test()

    elif flags_obj.mode == 'overall_test':

        cm.set_test_ui()
        vm.show_test_info(flags_obj)
        tester = PostTester(flags.FLAGS, None, cm, vm)
        tester.test()


if __name__ == "__main__":
    
    app.run(main)
