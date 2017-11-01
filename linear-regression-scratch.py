import mxnet.ndarray as nd
from mxnet import autograd

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2


X = nd.random_normal(shape=(num_examples,num_inputs))
y = true_w[0] * X[:,0] + true_w[1] * X[:,1] + true_b
y += .01 * nd.random_normal(shape = y.shape)
print(X[0],y[0])

import random
batch_size = 10

def data_iter():
    idx = list(range(num_examples))
