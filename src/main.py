import sys
import os
import numpy as np
import VNet as VN

cur_dir = os.path.dirname(os.path.abspath(__file__))

params = dict()
params['DataManagerParams'] = dict()
params['ModelParams'] = dict()

# params of the algorithm
params['ModelParams']['numcontrolpoints'] = 2
params['ModelParams']['sigma'] = 15
params['ModelParams']['device'] = 0
params['ModelParams']['prototxtTrain'] = os.path.join(cur_dir, '../models/train_noPooling_ResNet_cinque.prototxt')
params['ModelParams']['prototxtTest'] = os.path.join(cur_dir, '../models/test_noPooling_ResNet_cinque.prototxt')
params['ModelParams']['snapshot'] = 0
params['ModelParams']['dirTrain'] = os.path.join(cur_dir, '../data/train')
params['ModelParams']['dirTest'] = os.path.join(cur_dir, '../data/test')
params['ModelParams']['dirResult'] = os.path.join(cur_dir, '../results') # where we need to save the results (relative to the base path)
params['ModelParams']['dirSnapshots'] = os.path.join(cur_dir, '../snapshots/') # where to save the models while training
params['ModelParams']['batchsize'] = 2 # the batch size
params['ModelParams']['numIterations'] = 100000 # the number of iterations
params['ModelParams']['baseLR'] = 0.0001 # the learning rate, initial one
params['ModelParams']['nProc'] = 1 # the number of threads to do data augmentation

# params of the DataManager
params['DataManagerParams']['dstRes'] = np.asarray([1, 1, 1.5], dtype = float)
params['DataManagerParams']['VolSize'] = np.asarray([128, 128, 64], dtype = int)
params['DataManagerParams']['normDir'] = False # if rotates the volume according to its transformation in the mhd file.

model = VN.VNet(params)
train = [i for i, j in enumerate(sys.argv) if j == '-train']
if len(train) > 0:
    model.train()

test = [i for i, j in enumerate(sys.argv) if j == '-test']
if len(test) > 0:
    model.test()

