# coding: utf-8

# 例8.3

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


# 绘制分类策略函数
def draw(clf, kernel):
    # 在二维平面绘制样本点
    plt.scatter(x[:, 0], x[:, 1], c=y, s=8, cmap=plt.cm.Paired)

    # 绘制决策边界和最大间隔分离超平面
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    ax.contour(
        XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )

    # 标注支持向量
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=50,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    ax.set_title(str(clf))
    plt.xlabel('radius.mean(x1)')
    plt.ylabel('texture.mean(x2)')
    # 绘制到图片文件'kernel.png'
    plt.savefig(f'{kernel}')
    plt.show()


# 训练SVM分类模型
def SVM():
    # 核函数：线性核函数、多项式核函数、rbf高斯核函数、双曲正切核函数
    Kernel = ["linear", "poly", "rbf", "sigmoid"]

    for kernel in Kernel:
        # 初始化非线性核函数的gamma为默认值、初始化多项式核函数的维度为默认值3、初始化惩罚参数C为默认值1
        clf = SVC(kernel=kernel
                  , gamma="auto"
                  , degree=3
                  , C=1
                  ).fit(x_train, y_train)
        # SVM可视化
        draw(clf, kernel)
        # 不同核函数的SVM在测试集上的分类准确率和分类结果
        print("The test_accuracy under kernel %s is： %f" % (kernel, clf.score(x_test, y_test)))
        print("The test_result under kernel %s is： %s" % (kernel, clf.predict(x_test)[0:10]))


### 主函数
if __name__ == '__main__':
    data = load_breast_cancer()
    # 样本的特征集
    x = data.data
    # 样本的标签集
    y = data.target
    # 查看数据集描述
    print(data.DESCR)

    # 数据预处理
    x = StandardScaler().fit_transform(x)

    # 只取前两个特征值
    x = x[:, :2]


    # 随机划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    print("训练集样本数和标签类数:", x_train.shape)
    print("测试集样本数和标签类数:", x_test.shape)

    print("测试集中前10个样本的特征值：")
    print(x_test[0:10])
    print("测试集中前10个样本的标签：", y_test[0:10])

    # 用训练集训练模型，用测试集测试分类准确率
    SVM()
