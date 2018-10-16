# fashion_mnist_keras
	利用keras深度学习框架快速建立深度学习模型，对fashion_mnist数据集进行训练
	网络结构为：batchnomalization + 2conv + maxpooling + 3fc + dropout
	模型参数为：
	*conv1： filter深度64，filter_size 5x5, 全零填充
	*conv2： filter深度128，filter_size 5x5, 全零填充
	*pooling: filter_size 2x2
	*3fc: 128x64x10
	*dropout_rate: 0.35
	*optimizer: Adam, learning_rate=0.001

	测试集准确率：0.9228
