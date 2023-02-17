import logging
import os
import os.path as osp
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ray import tune
from torch.utils.data import Subset
import torch.distributed as dist
from cords.utils.config_utils import load_config_data
from cords.utils.dist_utils import init_dist, average_gradients
from cords.utils.data.data_utils import WeightedSubset
from cords.utils.data.data_utils import collate
from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, OLRandomDataLoader, \
    CRAIGDataLoader, GradMatchDataLoader, RandomDataLoader, SELCONDataLoader
from cords.utils.data.dataloader.SL.nonadaptive import FacLocDataLoader
from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.models import *
from cords.utils.data.data_utils.collate import *
import pickle
from torch.utils.data.sampler import *
import time

class ImagenetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, data_source, repeat_chunk=1):
        self.data_source = data_source
        self.indices = data_source.current_indices
        self.repeat = repeat_chunk


    def __iter__(self):
        rpt = 0
        while True:

            cur_len = len(self.indices)
            for i in self.indices:
                yield i
            rpt += 1
            if rpt == self.repeat:
                rpt = 0
                self.data_source.load_next_chunk()
                self.indices = self.data_source.current_indices

    def __len__(self):
        return len(self.data_source)


class TrainClassifier:
    def __init__(self, config_file_data):
        self.cfg = config_file_data
        results_dir = osp.abspath(osp.expanduser(self.cfg.train_args.results_dir))
        
        if self.cfg.dss_args.type != "Full":
            all_logs_dir = os.path.join(results_dir, self.cfg.setting,
                                        self.cfg.dss_args.type,
                                        self.cfg.dataset.name,
                                        str(self.cfg.dss_args.fraction),
                                        str(self.cfg.dss_args.select_every))
        else:
            all_logs_dir = os.path.join(results_dir, self.cfg.setting,
                                        self.cfg.dss_args.type,
                                        self.cfg.dataset.name)

        os.makedirs(all_logs_dir, exist_ok=True)
        # setup logger
        plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                            datefmt="%m/%d %H:%M:%S")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(plain_formatter)
        s_handler.setLevel(logging.INFO)
        self.logger.addHandler(s_handler)
        f_handler = logging.FileHandler(os.path.join(all_logs_dir, self.cfg.dataset.name + "_" +
                                                     self.cfg.dss_args.type + ".log"))
        f_handler.setFormatter(plain_formatter)
        f_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(f_handler)
        self.logger.propagate = False

    """
    ############################## Loss Evaluation ##############################
    """

    def model_eval_loss(self, data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.cfg.train_args.device), \
                                  targets.to(self.cfg.train_args.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    """
    ############################## Model Creation ##############################
    """

    def create_model(self):
        if self.cfg.model.architecture == 'RegressionNet':
            model = RegressionNet(self.cfg.model.input_dim)
        elif self.cfg.model.architecture == 'ResNet18':
            model = ResNet18(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'ResNet50':
            model = ResNet50(self.cfg.model.numclasses)        
        elif self.cfg.model.architecture == 'MnistNet':
            model = MnistNet()
        elif self.cfg.model.architecture == 'ResNet164':
            model = ResNet164(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNet':
            model = MobileNet(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNetV2':
            model = MobileNetV2(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNet2':
            model = MobileNet2(output_size=self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'HyperParamNet':
            model = HyperParamNet(self.cfg.model.l1, self.cfg.model.l2)
        elif self.cfg.model.architecture == 'ThreeLayerNet':
            model = ThreeLayerNet(self.cfg.model.input_dim, self.cfg.model.numclasses, 
	    self.cfg.model.h1, self.cfg.model.h2)
        elif self.cfg.model.architecture == 'LSTM':
            model = LSTMClassifier(self.cfg.model.numclasses, self.cfg.model.wordvec_dim, \
                 self.cfg.model.weight_path, self.cfg.model.num_layers, self.cfg.model.hidden_size)
        else:
            raise(NotImplementedError)
        
        model = model.to(self.cfg.train_args.device)
        if self.cfg.train_args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[self.cfg.train_args.local_rank])
        return model

    """
    ############################## Loss Type, Optimizer and Learning Rate Scheduler ##############################
    """

    def loss_function(self):
        if self.cfg.loss.type == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction='none')
            
        elif self.cfg.loss.type == "MeanSquaredLoss":
            criterion = nn.MSELoss()
            criterion_nored = nn.MSELoss(reduction='none')
        return criterion, criterion_nored

    def optimizer_with_scheduler(self, model):
        if self.cfg.optimizer.type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.cfg.optimizer.lr,
                                  momentum=self.cfg.optimizer.momentum,
                                  weight_decay=self.cfg.optimizer.weight_decay,
                                  nesterov=self.cfg.optimizer.nesterov)
        elif self.cfg.optimizer.type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.type == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.cfg.optimizer.lr)

        if self.cfg.scheduler.type == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.cfg.scheduler.T_max)
        elif self.cfg.scheduler.type == 'linear_decay':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.scheduler.stepsize, gamma=self.cfg.scheduler.gamma)
        elif self.cfg.scheduler.type == 'multi_step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(self.cfg.train_args.num_epochs / 3), int(self.cfg.train_args.num_epochs * 2 / 3)])
        else:
            scheduler = None
        return optimizer, scheduler

    @staticmethod
    def generate_cumulative_timing(mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing

    @staticmethod
    def save_ckpt(state, ckpt_path):
        torch.save(state, ckpt_path)

    @staticmethod
    def load_ckpt(ckpt_path, model, optimizer):
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        metrics = checkpoint['metrics']
        return start_epoch, model, optimizer, loss, metrics

    def count_pkl(self, path):
        if not osp.exists(path):
            return -1
        return_val = 0
        file = open(path, 'rb')
        while(True):
            try:
                _ = pickle.load(file)
                return_val += 1
            except EOFError:
                break
        file.close()
        return return_val

    def train(self):
        """
        ############################## Dist Set ##################
        """
        if self.cfg.train_args.distributed:
            rank, world_size = init_dist(backend=self.cfg.train_args.dist_backend)
            self.cfg.train_args.rank = rank
            self.cfg.train_args.world_size = world_size
            print("Distributed Enabled. Rank %d initalized" % self.cfg.train_args.rank)
        else:
            print("Single model training...")
        """
        ############################## General Training Loop with Data Selection Strategies ##############################
        """
        # Loading the Dataset
        logger = self.logger
        logger.info(self.cfg)
        if self.cfg.dataset.feature == 'classimb':
            trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
                                                               self.cfg.dataset.name,
                                                               self.cfg.dataset.feature,
                                                               classimb_ratio=self.cfg.dataset.classimb_ratio, dataset=self.cfg.dataset)
        else:
            trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
                                                               self.cfg.dataset.name,
                                                               self.cfg.dataset.feature, dataset=self.cfg.dataset)

        trn_batch_size = self.cfg.dataloader.batch_size
        val_batch_size = self.cfg.dataloader.batch_size
        tst_batch_size = self.cfg.dataloader.batch_size

        # import pdb
        # pdb.set_trace()


        if self.cfg.dataset.name == "sst2_facloc" and self.count_pkl(self.cfg.dataset.ss_path) == 1 and self.cfg.dss_args.type == 'FacLoc':
            self.cfg.dss_args.type = 'Full'
            file_ss = open(self.cfg.dataset.ss_path, 'rb')
            ss_indices = pickle.load(file_ss)
            file_ss.close()
            trainset = torch.utils.data.Subset(trainset, ss_indices)

        if 'collate_fn' not in self.cfg.dataloader.keys():
            collate_fn = None
        else:
            collate_fn = self.cfg.dataloader.collate_fn

        batch_sampler = lambda _, __ : None
        drop_last = False
        # if self.cfg.dataset.name in ['ilsvrc12']:
        #     batch_sampler = lambda dataset, bs : ImagenetSampler(dataset, repeat_chunk=1)

        if self.cfg.dss_args.type in ['SELCON']:
            drop_last = True
            assert(self.cfg.dataset.name in ['LawSchool_selcon', 'Community_Crime'])
            if self.cfg.dss_arg.batch_sampler == 'sequential':
                batch_sampler = lambda dataset, bs : torch.utils.data.BatchSampler(
                    torch.utils.data.SequentialSampler(dataset), batch_size=bs, drop_last=True
                )   # sequential
            elif self.cfg.dss_arg.batch_sampler == 'random':
                batch_sampler = lambda dataset, bs : torch.utils.data.BatchSampler(
                    torch.utils.data.RandomSampler(dataset), batch_size=bs, drop_last=True
                )   # random


        if self.cfg.dataset.name == "sst2_facloc" and self.count_pkl(self.cfg.dataset.ss_path) == 1 and self.cfg.dss_args.type == 'FacLoc':
            self.cfg.dss_args.type = 'Full'
            file_ss = open(self.cfg.dataset.ss_path, 'rb')
            ss_indices = pickle.load(file_ss)
            file_ss.close()
            trainset = torch.utils.data.Subset(trainset, ss_indices)

        if 'collate_fn' not in self.cfg.dataloader.keys():
            collate_fn = None
        else:
            collate_fn = self.cfg.dataloader.collate_fn

        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, sampler=batch_sampler(trainset, trn_batch_size),
                                                  shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last, num_workers=4)

        valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size, sampler=batch_sampler(validset, val_batch_size),
                                                shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last, num_workers=4)

        testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size, sampler=batch_sampler(testset, tst_batch_size),
                                                 shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last, num_workers=4)

        substrn_losses = list()  # np.zeros(cfg['train_args']['num_epochs'])
        trn_losses = list()
        val_losses = list()  # np.zeros(cfg['train_args']['num_epochs'])
        tst_losses = list()
        subtrn_losses = list()
        timing = list()
        trn_acc = list()
        val_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])
        tst_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])
        subtrn_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])

        # Checkpoint file
        checkpoint_dir = osp.abspath(osp.expanduser(self.cfg.ckpt.dir))
        ckpt_dir = os.path.join(checkpoint_dir, self.cfg.setting,
                                self.cfg.dss_args.type,
                                self.cfg.dataset.name,
                                str(self.cfg.dss_args.fraction),
                                str(self.cfg.dss_args.select_every))
        checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model Creation
        model = self.create_model()

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(model)

        """
        ############################## Custom Dataloader Creation ##############################
        """

        if 'collate_fn' not in self.cfg.dss_args:
                self.cfg.dss_args.collate_fn = None

        if self.cfg.dss_args.type in ['GradMatch', 'GradMatchPB', 'GradMatch-Warm', 'GradMatchPB-Warm']:
            """
            ############################## GradMatch Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device
            # by lys
            self.cfg.dss_args.distributed = self.cfg.train_args.distributed

            if self.cfg.train_args.distributed:
                dataloader = GradMatchDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                             batch_size=self.cfg.dataloader.batch_size//self.cfg.train_args.world_size,
                                             pin_memory=self.cfg.dataloader.pin_memory,
                                             collate_fn = self.cfg.dss_args.collate_fn)
                train_sampler = dataloader.train_sampler
            else:
                dataloader = GradMatchDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                             batch_size=self.cfg.dataloader.batch_size,
                                             shuffle=self.cfg.dataloader.shuffle,
                                             pin_memory=self.cfg.dataloader.pin_memory,
                                             collate_fn = self.cfg.dss_args.collate_fn)


        elif self.cfg.dss_args.type in ['GLISTER', 'GLISTER-Warm', 'GLISTERPB', 'GLISTERPB-Warm']:
            """
            ############################## GLISTER Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device
            dataloader = GLISTERDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                           batch_size=self.cfg.dataloader.batch_size,
                                           shuffle=self.cfg.dataloader.shuffle,
                                           pin_memory=self.cfg.dataloader.pin_memory,
                                           collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['CRAIG', 'CRAIG-Warm', 'CRAIGPB', 'CRAIGPB-Warm']:
            """
            ############################## CRAIG Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = CRAIGDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                         batch_size=self.cfg.dataloader.batch_size,
                                         shuffle=self.cfg.dataloader.shuffle,
                                         pin_memory=self.cfg.dataloader.pin_memory,
                                         collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['Random', 'Random-Warm']:
            """
            ############################## Random Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = RandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                          batch_size=self.cfg.dataloader.batch_size,
                                          shuffle=self.cfg.dataloader.shuffle,
                                          pin_memory=self.cfg.dataloader.pin_memory, 
                                          collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type == ['OLRandom', 'OLRandom-Warm']:
            """
            ############################## OLRandom Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = OLRandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,
                                            collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type == 'FacLoc':
            """
            ############################## Facility Location Dataloader Additional Arguments ##############################
            """
            wt_trainset = WeightedSubset(trainset, list(range(len(trainset))), [1] * len(trainset))
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.model = model
            self.cfg.dss_args.data_type = self.cfg.dataset.type
            
            dataloader = FacLocDataLoader(trainloader, valloader, self.cfg.dss_args, logger, 
                                          batch_size=self.cfg.dataloader.batch_size,
                                          shuffle=self.cfg.dataloader.shuffle,
                                          pin_memory=self.cfg.dataloader.pin_memory, 
                                          collate_fn = self.cfg.dss_args.collate_fn)
            if self.cfg.dataset.name == "sst2_facloc" and self.count_pkl(self.cfg.dataset.ss_path) < 1:

                ss_indices = dataloader.subset_indices
                file_ss = open(self.cfg.dataset.ss_path, 'wb')
                try:
                    pickle.dump(ss_indices, file_ss)
                except EOFError:
                    pass
                file_ss.close()

        elif self.cfg.dss_args.type == 'Full':
            """
            ############################## Full Dataloader Additional Arguments ##############################
            """
            wt_trainset = WeightedSubset(trainset, list(range(len(trainset))), [1] * len(trainset))
            # by lys
            if self.cfg.train_args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(wt_trainset)
                dataloader = torch.utils.data.DataLoader(wt_trainset, 
                                                     sampler=train_sampler, 
                                                     shuffle=(train_sampler is None),
                                                     batch_size=self.cfg.dataloader.batch_size//self.cfg.train_args.world_size,
                                                     pin_memory=self.cfg.dataloader.pin_memory,
                                                     collate_fn=self.cfg.dss_args.collate_fn,
                                                     num_workers=4
                                                     )
            else:
                dataloader = torch.utils.data.DataLoader(wt_trainset,
                                                     batch_size=self.cfg.dataloader.batch_size,
                                                     shuffle=self.cfg.dataloader.shuffle,
                                                     pin_memory=self.cfg.dataloader.pin_memory,
                                                     collate_fn=self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['SELCON']:
            """
            ############################## SELCON Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.lr = self.cfg.optimizer.lr
            self.cfg.dss_args.loss = criterion_nored # doubt: or criterion
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.optimizer = optimizer
            self.cfg.dss_args.criterion = criterion
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.batch_size = self.cfg.dataloader.batch_size
            
            # todo: not done yet
            self.cfg.dss_args.delta = torch.tensor(self.cfg.dss_args.delta)
            # self.cfg.dss_args.linear_layer = self.cfg.dss_args.linear_layer # already there, check glister init
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            
            dataloader = SELCONDataLoader(trainset, validset, trainloader, valloader, self.cfg.dss_args, logger,
                                           batch_size=self.cfg.dataloader.batch_size,
                                           shuffle=self.cfg.dataloader.shuffle,
                                           pin_memory=self.cfg.dataloader.pin_memory)

        else:
            raise NotImplementedError

        if self.cfg.dss_args.type in ['SELCON']:        
            is_selcon = True
        else:
            is_selcon = False


        """
        ################################################# Checkpoint Loading #################################################
        """

        if self.cfg.ckpt.is_load:
            start_epoch, model, optimizer, ckpt_loss, load_metrics = self.load_ckpt(checkpoint_path, model, optimizer)
            logger.info("Loading saved checkpoint model at epoch: {0:d}".format(start_epoch))
            for arg in load_metrics.keys():
                if arg == "val_loss":
                    val_losses = load_metrics['val_loss']
                if arg == "val_acc":
                    val_acc = load_metrics['val_acc']
                if arg == "tst_loss":
                    tst_losses = load_metrics['tst_loss']
                if arg == "tst_acc":
                    tst_acc = load_metrics['tst_acc']
                if arg == "trn_loss":
                    trn_losses = load_metrics['trn_loss']
                if arg == "trn_acc":
                    trn_acc = load_metrics['trn_acc']
                if arg == "subtrn_loss":
                    subtrn_losses = load_metrics['subtrn_loss']
                if arg == "subtrn_acc":
                    subtrn_acc = load_metrics['subtrn_acc']
                if arg == "time":
                    timing = load_metrics['time']
        else:
            start_epoch = 0

        """
        ################################################# Training Loop #################################################
        """

        for epoch in range(start_epoch, self.cfg.train_args.num_epochs):
            # by lys
            if self.cfg.train_args.distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            # print('Epoch {}'.format(epoch))
            subtrn_loss = 0
            subtrn_correct = 0
            subtrn_total = 0
            model.train()
            start_time = time.time()
            cum_weights = 0

            startt_time = time.time()
            for ii, data in enumerate(dataloader):                

                logging.info('The length of the current dataloader is: '.format(len(dataloader)))
                if ii % 100 == 0:
                    print('Epoch: {}, Iter: {}'.format(epoch, ii))    
                if is_selcon:
                    inputs, targets, _, weights = data  # dataloader also returns id in case of selcon algorithm
                else:
                    inputs, targets, weights = data
                    # print(inputs.shape, '======')
                    # print(targets.shape, '+++++++')
            
                inputs = inputs.to(self.cfg.train_args.device)
                targets = targets.to(self.cfg.train_args.device)
                weights = weights.to(self.cfg.train_args.device)
                

                optimizer.zero_grad()
                outputs = model(inputs)
                losses = criterion_nored(outputs, targets) / self.cfg.train_args.world_size

                if self.cfg.is_reg:
                    loss = torch.dot(losses.view(-1), weights / (weights.sum()))
                else:
                    loss = torch.dot(losses, weights / (weights.sum()))
                loss.backward()
                if self.cfg.train_args.distributed:
                    average_gradients(model)
                subtrn_loss += (loss.item() * weights.sum())
                cum_weights += weights.sum()
                optimizer.step()
                

                if not self.cfg.is_reg:
                    if is_selcon:
                        predicted = outputs     # linaer regression in selcon
                    else:
                        _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()

                end_time = time.time()
                # print('time spent:', end_time - startt_time, '========')
                
                # if ii >= len(trainloader):
                #     break 
            epoch_time = time.time() - start_time
            if cum_weights != 0:
                subtrn_loss = subtrn_loss/cum_weights
            if not scheduler == None:
                scheduler.step()
            timing.append(epoch_time)
            print_args = self.cfg.train_args.print_args

            """
            ################################################# Evaluation Loop #################################################
            """
            print('Evaluating...')
            if ((epoch + 1) % self.cfg.train_args.print_every == 0) or (epoch == self.cfg.train_args.num_epochs - 1):
                trn_loss = 0
                trn_correct = 0
                trn_total = 0
                val_loss = 0
                val_correct = 0
                val_total = 0
                tst_correct = 0
                tst_total = 0
                tst_loss = 0
                model.eval()

                if ("trn_loss" in print_args) or ("trn_acc" in print_args):
                    samples =0
                    with torch.no_grad():
                        for _, data in enumerate(trainloader):
                            if is_selcon:
                                inputs, targets, _ = data
                            else:
                                inputs, targets = data

                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets) / self.cfg.train_args.world_size
                            # by lys
                            reduced_loss = loss.data.clone()
                            if self.cfg.train_args.distributed:
                                dist.all_reduce(reduced_loss)
                            trn_loss += (reduced_loss.item() * trainloader.batch_size)
                            samples += targets.shape[0]
                            if "trn_acc" in print_args:
                                if is_selcon: predicted = outputs
                                else: _, predicted = outputs.max(1)
                                trn_total += targets.size(0)
                                # by lys
                                acc = predicted.eq(targets).sum().item()
                                reduced_acc = acc.clone() / self.cfg.train_args.world_size
                                if self.cfg.train_args.distributed:
                                    dist.all_reduce(reduced_acc)
                                trn_correct += reduced_acc
                                # trn_correct += predicted.eq(targets).sum().item()
                        trn_loss = trn_loss/samples
                        trn_losses.append(trn_loss)

                    if "trn_acc" in print_args:
                        trn_acc.append(trn_correct / trn_total)

                if ("val_loss" in print_args) or ("val_acc" in print_args):
                    samples =0
                    with torch.no_grad():
                        for _, data in enumerate(valloader):
                            if is_selcon:
                                inputs, targets, _ = data
                            else:
                                inputs, targets = data

                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += (loss.item() * valloader.batch_size)
                            samples += targets.shape[0]
                            if "val_acc" in print_args:
                                if is_selcon: predicted = outputs
                                else: _, predicted = outputs.max(1)
                                val_total += targets.size(0)
                                val_correct += predicted.eq(targets).sum().item()
                        val_loss = val_loss/samples
                        val_losses.append(val_loss)

                    if "val_acc" in print_args:
                        val_acc.append(val_correct / val_total)

                if ("tst_loss" in print_args) or ("tst_acc" in print_args):
                    samples =0
                    with torch.no_grad():
                        for jj, data in enumerate(testloader):
                            if is_selcon:
                                inputs, targets, _ = data
                            else:
                                inputs, targets = data

                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            tst_loss += (loss.item() * testloader.batch_size)
                            samples += targets.shape[0]
                            if "tst_acc" in print_args:
                                if is_selcon: predicted = outputs
                                else: _, predicted = outputs.max(1)
                                tst_total += targets.size(0)
                                tst_correct += predicted.eq(targets).sum().item()
                            # if jj >= len(testloader):
                            #     break 

                       

                        tst_loss = tst_loss/samples
                        tst_losses.append(tst_loss)

                    if "tst_acc" in print_args:
                        tst_acc.append(tst_correct / tst_total)

                if "subtrn_acc" in print_args:
                    subtrn_acc.append(subtrn_correct / subtrn_total)

                if "subtrn_losses" in print_args:
                    subtrn_losses.append(subtrn_loss)

                print_str = "Epoch: " + str(epoch + 1)

                """
                ################################################# Results Printing #################################################
                """

                for arg in print_args:

                    if arg == "val_loss":
                        print_str += " , " + "Validation Loss: " + str(val_losses[-1])

                    if arg == "val_acc":
                        print_str += " , " + "Validation Accuracy: " + str(val_acc[-1])

                    if arg == "tst_loss":
                        print_str += " , " + "Test Loss: " + str(tst_losses[-1])

                    if arg == "tst_acc":
                        print_str += " , " + "Test Accuracy: " + str(tst_acc[-1])

                    if arg == "trn_loss":
                        print_str += " , " + "Training Loss: " + str(trn_losses[-1])

                    if arg == "trn_acc":
                        print_str += " , " + "Training Accuracy: " + str(trn_acc[-1])

                    if arg == "subtrn_loss":
                        print_str += " , " + "Subset Loss: " + str(subtrn_losses[-1])

                    if arg == "subtrn_acc":
                        print_str += " , " + "Subset Accuracy: " + str(subtrn_acc[-1])

                    if arg == "time":
                        print_str += " , " + "Timing: " + str(timing[-1])

                # report metric to ray for hyperparameter optimization
                if 'report_tune' in self.cfg and self.cfg.report_tune  and len(dataloader):
                    tune.report(mean_accuracy=val_acc[-1])

                logger.info(print_str)

            """
            ################################################# Checkpoint Saving #################################################
            """

            if ((epoch + 1) % self.cfg.ckpt.save_every == 0) and self.cfg.ckpt.is_save:

                metric_dict = {}

                for arg in print_args:
                    if arg == "val_loss":
                        metric_dict['val_loss'] = val_losses
                    if arg == "val_acc":
                        metric_dict['val_acc'] = val_acc
                    if arg == "tst_loss":
                        metric_dict['tst_loss'] = tst_losses
                    if arg == "tst_acc":
                        metric_dict['tst_acc'] = tst_acc
                    if arg == "trn_loss":
                        metric_dict['trn_loss'] = trn_losses
                    if arg == "trn_acc":
                        metric_dict['trn_acc'] = trn_acc
                    if arg == "subtrn_loss":
                        metric_dict['subtrn_loss'] = subtrn_losses
                    if arg == "subtrn_acc":
                        metric_dict['subtrn_acc'] = subtrn_acc
                    if arg == "time":
                        metric_dict['time'] = timing

                ckpt_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': self.loss_function(),
                    'metrics': metric_dict
                }

                # save checkpoint
                self.save_ckpt(ckpt_state, checkpoint_path)
                logger.info("Model checkpoint saved at epoch: {0:d}".format(epoch + 1))

        """
        ################################################# Results Summary #################################################
        """

        logger.info(self.cfg.dss_args.type + " Selection Run---------------------------------")
        logger.info("Final SubsetTrn: {0:f}".format(subtrn_loss))
        if "val_loss" in print_args:
            if "val_acc" in print_args:
                logger.info("Validation Loss: %.2f , Validation Accuracy: %.2f", val_loss, val_acc[-1])
            else:
                logger.info("Validation Loss: %.2f", val_loss)

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                logger.info("Test Loss: %.2f, Test Accuracy: %.2f", tst_loss, tst_acc[-1])
            else:
                logger.info("Test Data Loss: %f", tst_loss)
        logger.info('---------------------------------------------------------------------')
        logger.info(self.cfg.dss_args.type)
        logger.info('---------------------------------------------------------------------')

        """
        ################################################# Final Results Logging #################################################
        """

        if "val_acc" in print_args:
            val_str = "Validation Accuracy: "
            for val in val_acc:
                if val_str == "Validation Accuracy: ":
                    val_str = val_str + str(val)
                else:
                    val_str = val_str + " , " + str(val)
            logger.info(val_str)

        if "tst_acc" in print_args:
            tst_str = "Test Accuracy: "
            for tst in tst_acc:
                if tst_str == "Test Accuracy: ":
                    tst_str = tst_str + str(tst)
                else:
                    tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

        if "time" in print_args:
            time_str = "Time: "
            for t in timing:
                if time_str == "Time: ":
                    time_str = time_str + str(t)
                else:
                    time_str = time_str + " , " + str(t)
            logger.info(time_str)

        omp_timing = np.array(timing)
        omp_cum_timing = list(self.generate_cumulative_timing(omp_timing))
        logger.info("Total time taken by %s = %.4f ", self.cfg.dss_args.type, omp_cum_timing[-1])
