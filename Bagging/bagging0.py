import numpy as np
from sklearn import datasets 
#用于多样性度量,采用马修斯相关系数
from sklearn.metrics import matthews_corrcoef

from sklearn.model_selection import train_test_split
#评分
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import BaggingClassifier

#引入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#引入SVC分类器
from sklearn.svm import SVC

#引入KNN分类器
from sklearn.neighbors import KNeighborsClassifier

#选用葡萄酒数据集，样本数178,属性数13
X, y = datasets.load_wine(return_X_y=True)
#print(X.shape)
#print(y.shape)

#决策树的训练效果

#划分数据为测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, 
                                    random_state = 42)

#决策树的baging方法，每次选取50个样本，一共100个基分类器
bag_tree = BaggingClassifier( DecisionTreeClassifier(), n_estimators=100, max_samples=50, bootstrap=True, n_jobs=-1 )

bag_tree.fit(X_train, y_train)
#十次交叉验证，取平均值, 0.95
bag_tree_score = cross_val_score(bag_tree, X_test, y_test, cv = 10)
print('the out-of-bag-accuary of bagging tree is ', np.mean(bag_tree_score))

#样本被选择的状态
choosen_state0 = bag_tree.estimators_samples_
#print(len(choosen_state))，　
#print(choosen_state[1].shape)，　

#计算每次选样的相关性，数值越小说明相关性小，多样性比较好,0.99
metrix0 = 0
for i in range(len(choosen_state0)):
    for j in range(len(choosen_state0)):
        metrix0+= matthews_corrcoef(choosen_state0[i], choosen_state0[j])
print('the metrix of bagging-tree is ', 1 - metrix0/(len(choosen_state0)**2))

#SVC bagging
bag_svc = BaggingClassifier(SVC(kernel = 'rbf',gamma= 'auto'), n_estimators = 100, max_samples=50, bootstrap=True, n_jobs=-1)
bag_svc.fit(X_train, y_train)
bag_svc_score = cross_val_score(bag_svc, X_test, y_test, cv = 10)

print('the out-of-bag-accuary of bagging svc is ', np.mean(bag_svc_score))

#计算每次选样的相关性，数值越小说明相关性小，多样性比较好,0.99
choosen_state1 = bag_svc.estimators_samples_
metrix1 = 0
for i in range(len(choosen_state1)):
    for j in range(len(choosen_state1)):
        metrix1 += matthews_corrcoef(choosen_state1[i], choosen_state1[j])
print('the metrix of bagging-svc is ', 1 - metrix1/(len(choosen_state1)**2))


#knn bagging 
bag_knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=3), n_estimators=100, max_samples=50, bootstrap=True, n_jobs=-1)
bag_knn.fit(X_train, y_train)
bag_knn_score = cross_val_score(bag_knn, X_test, y_test, cv=10)

print('the out-of-bag-accuary of bagging knn is ', np.mean(bag_knn_score))

choosen_state2 = bag_knn.estimators_samples_

#计算每次选样的相关性，数值越小说明相关性小，多样性比较好,0.99
metrix2 = 0
for i in range(len(choosen_state2)):
    for j in range(len(choosen_state2)):
        metrix2 += matthews_corrcoef(choosen_state2[i], choosen_state2[j])
print('the metrix of bagging-knn is ', 1 - metrix2/(len(choosen_state2)**2))


