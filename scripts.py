import cPickle as pkl
import gzip
import os, socket, shutil
import sys, re
import time
from collections import OrderedDict
import numpy
import tables
import theano
import theano.tensor as T
import common

from multiprocessing import Process, Queue, Manager

dataset_path = common.get_rab_dataset_base_path()+'youtube2text_iccv15/'
CAP = common.load_pkl(dataset_path + 'CAP.pkl')
test_path = common.get_rab_exp_path()+'arctic-capgen-vid/test_non/'
test = open(test_path + 'test_samples.txt')

f = open(test_path + 'test_result.txt', 'w')
test_caps = test.readlines()

for i in range(670):
	ground_truth = CAP['vid' + str(1301+i)]
	f.write('vid' + str(1301+i) + '\n')
	for g in ground_truth:
		f.write('cap_id : ' + g['cap_id'] + '\tcaption : ' + g['caption'] + '\n')

	f.write('sample result : ' + test_caps[i] + '\n')

f.close()