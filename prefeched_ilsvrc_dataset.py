import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import *
import os
import math
import time
import multiprocessing
from PIL import Image
import numpy as np
import h5py
import random


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as img_file:
        with Image.open(img_file) as cur_img:
            img = cur_img.convert("RGB")
            cur_img.close()
        img_file.close()

    return img

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        print("Use accimage")
        return accimage_loader(path)
    else:
        return pil_loader(path)



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





class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        split='train',
        prefetch=True,
        num_workers=100,
        loader=default_loader,
        **kwargs
    ):
        root = self.root = os.path.expanduser(root)
        self.split = self._verify_split(split)

        super(ImageNet, self).__init__(self.split_folder, **kwargs)

        self.root = root
        self.loader = loader

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.split == 'train':
            self.transform = transforms.Compose([
                 transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize])

        else:
            assert self.split == 'val'
            self.transform = transforms.Compose([
                 transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 normalize])

        self.prefetch = prefetch
        self.prefetched_samples = None
        self.prefetched_targets = None
        self.chunk_datadir = []
        self.chunk_indices = []


        if self.prefetch:
            # import h5py

            print("Prefetching ImageNet...")
            prefetch_chunk = 1300000
            if split == "train":
                h5f_dir = root + "/h5f_whole/"
            else:
                h5f_dir = root + "/h5f_val_whole/"
            for steps in range(math.ceil(len(self) / prefetch_chunk)):
                print('The current step is: {}'.format(steps))
                h5f_chunk_file = h5f_dir + "chunk_" + str(steps)
                start_file = steps * prefetch_chunk
                end_file = min((steps + 1) * prefetch_chunk, len(self))
                self.chunk_datadir.append(h5f_chunk_file)
                self.chunk_indices.append(list(range(start_file, end_file)))
                if os.path.exists(h5f_chunk_file):
                    print("%s already exist, continue..." % (h5f_chunk_file))
                    continue
                t0 = time.time()
                print(
                    "==== Fetching chunk %d: %d to %d ======"
                    % (steps, start_file, end_file)
                )
                async_samples = multiprocessing.Manager().list(
                    [None] * (end_file - start_file)
                )
                async_targets = multiprocessing.Manager().list(
                    [None] * (end_file - start_file)
                )
                fetch_jobs = []
                for i in range(num_workers):
                    p = multiprocessing.Process(
                        target=self.fetch_data_worker,
                        args=(i, num_workers, async_samples, async_targets, start_file),
                    )
                    fetch_jobs.append(p)
                    p.start()
                for p in fetch_jobs:
                    p.join()

                sample_data_chunk = np.stack(async_samples, axis=0)
                target_data_chunk = np.array(async_targets)
                h5file_towrite = h5py.File(h5f_chunk_file, "w")
                h5file_towrite.create_dataset("samples", data=sample_data_chunk)
                h5file_towrite.create_dataset("targets", data=target_data_chunk)
                h5file_towrite.close()
                print(
                    "%s created using %f seconds" % (h5f_chunk_file, time.time() - t0)
                )

            self.chunk_idx = -1
            self.chunk_sequence = list(range(len(self.chunk_datadir)))
            self.load_next_chunk()



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.prefetch:
            sample = self.prefetched_samples[index % self.current_chunk_size]
            target = self.prefetched_targets[index % self.current_chunk_size]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, torch.tensor(target)





    def load_next_chunk(self):
        num_chunks = len(self.chunk_datadir)
        old_chunk_idx = self.chunk_idx
        chunk_seq_idx = self.chunk_idx + 1
        if chunk_seq_idx >= num_chunks:
            random.shuffle(self.chunk_sequence)
            print("Current chunk sequence:", self.chunk_sequence)
        self.chunk_idx = self.chunk_sequence[chunk_seq_idx % num_chunks]

        self.current_indices = self.chunk_indices[self.chunk_idx]
        self.current_chunk_size = len(self.current_indices)

        self.start_index = self.current_indices[0]
        if old_chunk_idx != self.chunk_idx or self.prefetched_samples is None:
            print(
                "Loading chunk %d (%d to %d)..."
                % (self.chunk_idx, self.current_indices[0], self.current_indices[-1])
            )
            t0 = time.time()
            cur_chunk_data = self.chunk_datadir[self.chunk_idx]
            cur_h5f = h5py.File(self.chunk_datadir[self.chunk_idx], "r")
            # clear mem first
            self.prefetched_samples = None
            self.prefetched_targets = None
            self.prefetched_samples = torch.tensor(cur_h5f["samples"][:])
            self.prefetched_targets = torch.tensor(cur_h5f["targets"][:])
            print("finished with %f seconds" % (time.time() - t0), flush=True)


    def fetch_data_worker(
        self, proc_id, num_procs, prefetched_samples, prefetched_targets, start_file
    ):
        data_len = len(prefetched_samples)
        chunk_size = math.ceil(data_len / num_procs)
        start_id = proc_id * chunk_size + start_file
        end_id = min((proc_id + 1) * chunk_size, data_len) + start_file
        print("Worker %d: %d to %d" % (proc_id, start_id, end_id))
        n_done = 0
        t0 = time.time()
        for index in range(start_id, end_id):
            # print('The index is:', index)
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            prefetched_samples[index - start_file] = sample.numpy()
            prefetched_targets[index - start_file] = target
            n_done += 1
            if n_done % 100 == 0 and proc_id == 0:
                print("Worker %d finished %d/%d" % (proc_id, n_done, end_id - start_id))
        if proc_id == 0:
            print(
                "Worker %d finished %d images in %f s"
                % (proc_id, n_done, time.time() - t0)
            )


    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)


    @property
    def valid_splits(self):
        return "train", "val"


if __name__ == '__main__':
    train_set = ImageNet(root = '~/Projects/data_selection/datasets/ilsvrc12/imagenet', split='train')
    test_set = ImageNet(root = '~/Projects/data_selection/datasets/ilsvrc12/imagenet', split='val')

    batch_sampler = lambda dataset, bs : ImagenetSampler(dataset, repeat_chunk=1)

    # train_loader = ImagenetSampler(train_set, repeat_chunk=1)
    # test_loader = ImagenetSampler(test_set, repeat_chunk=1)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=256, sampler=batch_sampler(train_set, 256),
                                              shuffle = False, pin_memory = True, drop_last = True, num_workers = 1)

    # testloader = torch.utils.data.DataLoader(test_set, batch_size=256, sampler=batch_sampler(test_set, 256),
    #                                          shuffle=False, pin_memory=True, drop_last=True, num_workers=1)
    
    for epoch in range(5):    
        for i, (x, y) in enumerate(trainloader): 
            print(i)

            if i >= len(trainloader):
                break 
