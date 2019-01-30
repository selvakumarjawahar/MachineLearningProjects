import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import struct
import sys
import urllib.request as req 
import imageio as imgio 

def loadData(src, cimg):
    print ('Downloading ' + src)
    gzfname, _ = req.urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))
def loadLabels(src, cimg):
    print ('Downloading ' + src)
    gzfname, _ = req.urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, 1))

def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))

def savetxt(filename, ndarray):
    with open(filename, 'w') as f:
        labels = list(map(' '.join, np.eye(10, dtype=np.uint8).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))

def savemystery(filename):
    with open('mystery.txt','w') as f_write:
        img_arr = imgio.imread(filename)
        img_str = img_arr.tostring()
        f_write.write('|labels {} |features {}\n'.format('0010000000',img_str))


if __name__ == '__main__':
    #url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    #url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    #num_train_samples = 60000

    #url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    #url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    #num_test_samples = 10000

    #train = try_download(url_train_image, url_train_labels, num_train_samples)
    #test = try_download(url_test_image, url_test_labels, num_test_samples)
    #savetxt("Train-28x28_cntk_text.txt", train)
    #savetxt("Test-28x28_cntk_text.txt", test)
    savemystery('MysteryNumberD.bmp')
    print('Done')
