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
from sklearn.metrics import matthews_corrcoef
import numpy as np

X = np.array([True, False])
print(int(len(X)))