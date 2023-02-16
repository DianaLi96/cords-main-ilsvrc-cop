from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data
import sys
import os

# os.environ['CUDA_VISIBLE_DEVICES']='1'

config_file = "./configs/SL/config_full_ilsvrc.py" #'./configs/SL/config_gradmatchpb-warm_imagenet.py'
cfg = load_config_data(config_file)
clf = TrainClassifier(cfg)
clf.train()
