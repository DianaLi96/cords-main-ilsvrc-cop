from abc import abstractmethod
from cords.utils.data.data_utils import WeightedSubset
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np


# Base objects
class DSSDataLoader:
    """
    Implementation of DSSDataLoader class which serves as base class for dataloaders of other
    selection strategies for supervised learning framework.

    Parameters
    -----------
    full_data: torch.utils.data.Dataset Class
        Full dataset from which data subset needs to be selected.
    dss_args: dict 
        Data subset selection arguments dictionary
    logger: class
        Logger class for logging the information
    """
    def __init__(self, full_data, dss_args, logger, *args, **kwargs):
        """
        Constructor Method
        """
        super(DSSDataLoader, self).__init__()
        # TODO: Integrate verbose in logging
        self.len_full = len(full_data)
        assert "fraction" in dss_args.keys(), "'fraction' is a compulsory argument. Include it as a key in dss_args"
        if (dss_args.fraction > 1) or (dss_args.fraction<0):
             raise ValueError("'fraction' should lie between 0 and 1")

        self.fraction = dss_args.fraction
        self.budget = int(self.len_full * self.fraction)
        self.logger = logger
        self.dataset = full_data
        self.loader_args = args
        self.loader_kwargs = kwargs
        self.subset_indices = None
        self.subset_weights = None
        self.subset_loader = None
        self.batch_wise_indices = None
        #self.strategy = None
        self.cur_epoch = 1
        self.distributed = dss_args.distributed
        wt_trainset = WeightedSubset(full_data, list(range(len(full_data))), [1]*len(full_data))
        # by lys
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(wt_trainset)
            self.wtdataloader = torch.utils.data.DataLoader(wt_trainset, 
                                                     sampler=self.train_sampler, 
                                                     shuffle=(self.train_sampler is None),
                                                     num_workers=4,
                                                     *self.loader_args,
                                                     **self.loader_kwargs
                                                     )
            
        else:
            self.train_sampler = None
            self.wtdataloader = torch.utils.data.DataLoader(wt_trainset, *self.loader_args, **self.loader_kwargs)
        self._init_subset_loader()

    def __getattr__(self, item):
        return object.__getattribute__(self, "subset_loader").__getattribute__(item)

    def _init_subset_loader(self):
        """
        Function that initializes the random data subset loader
        """
        # All strategies start with random selection
        self.subset_indices = self._init_subset_indices()
        self.subset_weights = torch.ones(self.budget)
        self._refresh_subset_loader()

    # Default subset indices comes from random selection
    def _init_subset_indices(self):
        """
        Function that initializes the subset indices randomly
        """
        return np.random.choice(self.len_full, size=self.budget, replace=False)

    def _refresh_subset_loader(self):
        """
        Function that regenerates the data subset loader using new subset indices and subset weights
        """
        if self.distributed:
            wt_dataset = WeightedSubset(self.dataset, self.subset_indices, self.subset_weights)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(wt_dataset)
            self.subset_loader = torch.utils.data.DataLoader(wt_dataset, 
                                                     sampler=self.train_sampler, 
                                                     shuffle=(self.train_sampler is None),
                                                     num_workers=4,
                                                     *self.loader_args,
                                                     **self.loader_kwargs
                                                     )
        else:
            self.subset_loader = DataLoader(WeightedSubset(self.dataset, self.subset_indices, self.subset_weights), 
                                            *self.loader_args, **self.loader_kwargs)
        
        self.batch_wise_indices = list(self.subset_loader.batch_sampler)

