import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)






def folder2lmdb(dpath, name="train", write_frequency=5):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    if name=="train":
        transform_ilsvrc = transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          # transforms.ToTensor(),
          # normalize 
          ])
    else:
        transform_ilsvrc = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # normalize 
        ])

    dataset = ImageFolder(directory, transform=transform_ilsvrc, loader=raw_reader)
    

    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)
    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    # print('1234')
    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        # print('Here now!', idx)
        image, label = data[0]
        # print(type(image))
        # print(type(label))
        # print('image is of shape:', image.shape)
        # print('label is of shape:', label.shape)

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((image, label)))

        # txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    # generate lmdb
    folder2lmdb('/export/home/personal/wangjiaxing/Projects/data_selection/datasets/ilsvrc12/imagenet', name='train')
    folder2lmdb('/export/home/personal/wangjiaxing/Projects/data_selection/datasets/ilsvrc12/imagenet', name='val')
