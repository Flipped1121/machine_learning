import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

# 获取数据
data = pd.read_csv(r"D:\Python_Data\web\validate_1000.csv")

# 缺失值处理
data = data.fillna(data.mean())

# 是否存在缺失值
# print(data.isnull().any())
# print(data)

y = data["label"]
x = data.iloc[:, 1:-1]

test = pd.read_csv("validate_1000.csv")
test = test.fillna(test.mean())

y1 = test["label"]
x1 = test.iloc[:, 1:-1]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
# # 1）实例化一个转换器类
# transfer = PCA(n_components=0.95)
# # 2）调用fit_transform 降维处理
# data_new = transfer.fit_transform(data)
# 3）标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x1 = transfer.transform(x1)

# # 加入网格搜索与交叉验证
# estimator = RandomForestClassifier()
#
# # 参数准备
# param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
# estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
# estimator.fit(x_train, y_train)
#
# # 保存模型
# joblib.dump(estimator, 'FR_model.joblib')

# 加载模型
estimator = joblib.load('FR_model.joblib')

# 模型评估
# 直接比对真实值和预测值
y_predict = estimator.predict(x1)
print("y_predict:\n", y_predict)
print("随机森林")
print(classification_report(y_predict, y1))

# 计算准确率
score = estimator.score(x1, y1)
print("准确率为：\n", score)

# 最佳参数：best_params_
print("最佳参数：\n", estimator.best_params_)
# 最佳结果：best_score_
print("最佳结果：\n", estimator.best_score_)
# 最佳估计器：best_estimator_
print("最佳估计器:\n", estimator.best_estimator_)
# 交叉验证结果：cv_results_
print("交叉验证结果:\n", estimator.cv_results_)
