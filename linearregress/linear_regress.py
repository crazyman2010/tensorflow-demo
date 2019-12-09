import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# 线性回归， f(x) = 2*x + 1
def f(x):
    return 2 * x + 1.0


# 生成训练数据
# 使用np生成均匀生布于[-1,1]中的x训练数据
np.random.seed(4)
x_train = np.linspace(-1, 1, 100)
# 计算y训练数据（使用f(x)计算并增加干扰）
y_train = f(x_train) + np.random.randn(*x_train.shape) * 0.4

# 定义模型， 使用一层全连接， 输出为一维，输入为一维
model = tf.keras.Sequential(
    tf.keras.layers.Dense(1, input_shape=(1,))
)
# 编译模型， 使用adam(梯度下降）作为优化函数， 使用mse(均方差）作为损失函数
model.compile(optimizer='adam', loss='mse')

# 开始训练
model.fit(x_train, y_train, batch_size=10, verbose=2, epochs=800)

# 生成标签数据，评估误差
x_test = np.linspace(-1, 1, 30)
y_test = f(x_test)
model.evaluate(x_test, y_test)

# 使用模型计算x_test对应的y值
y_predict = model.predict(x_test)

# 画出 训练数据， f(x) 和 模型预测的值
plt.plot(x_test, y_test, color='red', linewidth=1)
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_predict, edgecolors='yellow')
plt.show()
