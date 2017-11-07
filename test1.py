import mxnet.ndarray as nd
import mxnet.autograd as ag
import numpy as np

# def f(a):
#     b = a * 2
#     while nd.norm(b).asscalar() < 1000:
#         b = b * 2
#     if nd.sum(b).asscalar() > 0:
#         c = b
#     else:
#         c = 100 * b
#     return c
#
# a = nd.random_normal(shape=3)
# a.attach_grad()
# with ag.record():
#     c = f(a)
# c.backward()
#
# print(a.grad == c/a)

x = nd.array([1])
x.attach_grad()

with ag.record():
    #y = x * x * 2 + x + 1
    y = nd.exp(x)

y.backward()
print(x.grad)

