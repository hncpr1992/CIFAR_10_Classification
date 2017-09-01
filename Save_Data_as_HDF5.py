import numpy as np
from copy import deepcopy
import pickle
import time
import h5py

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


folder = 'cifar-10-batches-py/'
#4 categories: ['data', 'labels', 'batch_label', 'filenames']
#first 1024 entries are red channel, next 1024 are green, and next 1024 are blue.  
#stored in row-major order.  first 32 entries of the array are the red channel values of the first row
#of the image

#load all training data into memory
for i in range(1,6):
	batch_dict = unpickle(folder+'data_batch_' + str(i))
	current_array_X = np.array(batch_dict['data'])
	current_array_Y = np.array(batch_dict['labels'])
	if (i== 1):
		X = current_array_X
		Y = current_array_Y
	else:
		X = np.concatenate((X, current_array_X))
		Y = np.concatenate((Y, current_array_Y))


#size of features
L_X = len(X[0,:])
#size of training data
L_Y = len(Y)


#normalize X (without normalization, it won't be able to train well !!!!
X = X/255.0

#reshape X
X = np.reshape(X, (L_Y,3,32,32))

#change to float32 and int32
X_train = np.float32(X)
Y_train = np.int32(Y)

#do the same procedure with test data
batch_dict = unpickle(folder+'test_batch')
X_test = np.array(batch_dict['data'])
Y_test = np.array(batch_dict['labels'])
L_Y_test = len(Y_test)

X_test = X_test/255.0

X_test = np.reshape(X_test, (L_Y_test,3,32,32))

X_test = np.float32(X_test)
Y_test = np.int32(Y_test)

#save as HDF5 files
f = h5py.File('CIFAR10' + '.hdf5', 'w')  
f.create_dataset('X_train', data = X_train, compression = "gzip")
f.create_dataset('Y_train', data = Y_train, compression = "gzip")
f.create_dataset('X_test', data = X_test, compression = "gzip")
f.create_dataset('Y_test', data = Y_test, compression = "gzip")

f.close()

