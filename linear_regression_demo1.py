from mxnet import nd,autograd,gluon
import matplotlib.pyplot as plt
import mxnet as mx

num_inputs = 2
num_outputs = 1
num_examples = 10000

#noise = 0.01


data_ctx = mx.cpu()
model_ctx = mx.cpu()



X = nd.random_normal(shape=(num_examples,num_inputs),ctx = data_ctx)


def real_fn(X):
    return 2 * X[:,0] + 3 * X[:,1] + 4

noise = 0.01
#noise = .1 * nd.random_normal(shape=(num_examples,), ctx=data_ctx)
y = real_fn(X) + noise


#plt.scatter(X[:,1].asnumpy(),y.asnumpy())
#plt.show()

#取得数据
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X,y),batch_size = batch_size,shuffle=True)

# for i,(data,label) in enumerate(train_data):
#     print(data,label)
#     break


#模型参数

w = nd.random_normal(shape=(num_inputs,num_outputs),ctx = model_ctx)
b = nd.random_normal(shape=(num_outputs),ctx = model_ctx)
params = [w,b]

for param in params:
    param.attach_grad()

#模型
def net(X):
    return mx.nd.dot(X,w) + b

#loss
def square_loss(yhat,y):
    return nd.mean((yhat -y ) ** 2)

#optimizer
def SGD(params,lr):
    for param in params:
        param[:] = param - lr * param.grad


############################################
#    Script to plot the losses over time
############################################
def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()

    plt.show()


epochs = 10
learning_rate = .0001

smoothing_constant = .01
moving_loss = 0
niter = 0
losses = []
#plot(losses, X)

# for e in range(epochs):
#     for i, (data,label) in enumerate(train_data):
#         data = data.as_in_context(model_ctx)
#         label = label.as_in_context(model_ctx).reshape((-1,1))
#
#         with autograd.record():
#             output = net(data)
#             loss = square_loss(output,label)
#         loss.backward()
#         SGD(params,learning_rate)
#         cumulative_loss = loss.asscalar()
#     print(cumulative_loss/num_examples)

w[:] = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b[:] = nd.random_normal(shape=num_outputs, ctx=model_ctx)

for e in range(epochs):
    for i,(data,label) in enumerate(train_data):

        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output,label)
        loss.backward()

        SGD(params,learning_rate)

    #     cumulative_loss = loss.asscalar()
    # print(cumulative_loss/num_examples)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter +=1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correct the bias from the moving averages
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)

        if (i + 1) % 500 == 0:
            print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i , est_loss))
            losses.append(est_loss)

    #plot(losses, X)












