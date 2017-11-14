from mxnet import nd
def pure_batch_norm(X, gamma, beta, eps=1e-5):
    assert len(X.shape) in (2, 4)
    # 全连接: batch_size x feature
    if len(X.shape) == 2:
        # 每个输入维度在样本上的平均和方差
        mean = X.mean(axis=0)
        variance = ((X - mean)**2).mean(axis=0)
    # 2D卷积: batch_size x channel x height x width
    else:
        # 对每个通道算均值和方差，需要保持4D形状使得可以正确地广播
        mean = X.mean(axis=(0,2,3), keepdims=True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)

    # 均一化
    X_hat = (X - mean) / nd.sqrt(variance + eps)
    # 拉升和偏移
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)


#A = nd.arange(6).reshape((3,2))
#pure_batch_norm(A, gamma=nd.array([1,1]), beta=nd.array([0,0]))


B = nd.arange(18).reshape((1,2,3,3))
pure_batch_norm(B, gamma=nd.array([1,1]), beta=nd.array([0,0]))



