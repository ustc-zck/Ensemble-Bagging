import numpy as np
from sklearn import datasets 

#用相关性系数进行度量
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import BaggingClassifier

#引入多层感知机分类器，浅层神经网络
from sklearn.neural_network import MLPClassifier

#引入SVC分类器
from sklearn.svm import SVC

#引入朴素贝叶斯分类器，选择了高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB

#选用iris数据集
X, y = datasets.load_iris(return_X_y=True)

#print(X.shape)
#print(y.shape)

#划分数据为测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, 
                                    random_state = 42)


#多层感知机bagging
bag_mlp = BaggingClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5,
                                          hidden_layer_sizes=(5, 2), random_state=1))

bag_mlp.fit(X_train, y_train)

bag_mlp_score = bag_mlp.score(X_test, y_test)

print('the out-of-bag-accuary of bagging mlp is ', np.mean(bag_mlp_score))

#计算每次选样的相关性，数值越小说明相关性小，多样性比较好,0.99
choosen_state0 = bag_mlp.estimators_samples_
metrix0 = 0
for i in range(len(choosen_state0)):
    for j in range(len(choosen_state0)):
        metrix0 += matthews_corrcoef(choosen_state0[i], choosen_state0[j])
print('the metrix of bagging-mlp is ', 1 - metrix0/(len(choosen_state0)**2))

#SVC bagging
bag_svc = BaggingClassifier(SVC(kernel = 'rbf',gamma= 'auto'), n_estimators = 100, max_samples=50, bootstrap=True, n_jobs=-1)
bag_svc.fit(X_train, y_train)
bag_svc_score = bag_svc.score(X_test, y_test)

print('the out-of-bag-accuary of bagging svc is ', np.mean(bag_svc_score))

#计算每次选样的相关性，数值越小说明相关性小，多样性比较好,0.99
choosen_state1 = bag_svc.estimators_samples_
metrix1 = 0
for i in range(len(choosen_state1)):
    for j in range(len(choosen_state1)):
        metrix1 += matthews_corrcoef(choosen_state1[i], choosen_state1[j])
print('the metrix of bagging-svc is ', 1 - metrix1/(len(choosen_state1)**2))

#S朴素贝叶斯bagging

bag_nb = BaggingClassifier(GaussianNB(),n_estimators = 100, max_samples=50, bootstrap=True, n_jobs=-1)
bag_nb.fit(X_train, y_train)
bag_nb_score = bag_nb.score(X_test, y_test)

print('the out-of-bag-accuary of bagging nb is ', np.mean(bag_nb_score))

#计算每次选样的相关性，数值越小说明相关性小，多样性比较好,0.99
choosen_state2 = bag_nb.estimators_samples_
metrix2 = 0
for i in range(len(choosen_state2)):
    for j in range(len(choosen_state2)):
        metrix2 += matthews_corrcoef(choosen_state2[i], choosen_state2[j])
print('the metrix of bagging-nb is ', 1 - metrix1/(len(choosen_state2)**2))


