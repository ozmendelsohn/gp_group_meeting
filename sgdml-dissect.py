import sys
import numpy as np
from sgdml.train import GDMLTrain
from sgdml.utils import ui
import sgdml
import time


dataset = np.load('ethanol_dft.npz')
n_train = 100
use_torch = True
print( f'use_torch: {use_torch}')
gdml_train = GDMLTrain(use_torch=use_torch)
t0 = time.time()
task = gdml_train.create_task(dataset, n_train,
        valid_dataset=dataset, n_valid=100,
        sig=10, lam=1e-15, use_sym=False, solver='cg')
model = gdml_train.train(task, callback=ui.callback)
np.savez_compressed('m_ethanol.npz', **model)
print(f'-------------finished!------------\n Time elapsed: {time.time()-t0} [sec]')