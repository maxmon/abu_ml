
# 泰坦尼克号生存预测（下）——决策树和集成学习


作者：maxmon

官网：[www.abuquant.com](http://www.abuquant.com)

阿布量化版权所有 未经允许 禁止转载

> TAG: Machine Learning, 机器学习, 决策树, 随机森林, RF, GBDT, 泰坦尼克号, 集成学习, 模型融合, 非线性模型

____

- 信息基础理论
- 决策树
- 集成学习（模型融合）

## 环境

- [Anaconda主页](https://www.anaconda.com/download/) 下载终端运行Anaconda脚本
- [abupy量化系统主页](https://github.com/bbfamily/abu) 终端运行`pip install abupy`
- [泰坦尼克号数据集](https://github.com/maxmon/abu_ml/tree/master/ipython/data/titanic)下载数据集，为了实验方便，我们只选择train.csv作为全部实验数据集

## 信息基础理论

给定一个标定类别的数据集X，如何计算出数据集中信息量的大小呢？香农给出了一套基于概率计算信息量长度的公式：


```python
import numpy as np

def entropy(P):
    """根据每个样本出现的概率，计算信息量，输入P是数据集上每个数值统计的频率（概率）向量"""
    return -np.sum(P * np.log2(P))
```

让我们通过一个小故事理解上面的计算方式

    考场上，卷面有100道考题，每个题目有ABCD四个选项，出题老师提前说明每个选项都可能出现。我在给死党小明做小抄传答案，希望小抄尽可能短，不容易被监考老师抓到。而我其实是一个隐藏在人类世界的机器人，不会写字符语言如“A、B、C、D”，只会写0、1两种数字（用二进制位编码事件）。那么我应该如何传递答案，才能让小抄最短呢？
    
如果依然让我传递ABCD的ASCII码的话，每个答案需要占8位，如“01100001”表示“A”。从传输的角度，这显然很浪费，一些如“E、F”等后续的字母根本用不上。信息学最初要解决的问题，就是数据的压缩和传输。编码4个答案选项其实只需用2bit（位）就可以了：00-A,01-B,10-C,11-D。

    我传给小明的小抄内容：10110001……1011（CDAB……CD）。
    
用香农公式计算一下这个小抄的信息量：由于每个选项等可能出现，所以P(A)=P(B)=P(C)=p(D)=1/4。小抄的信息量是H(X) = A(1/4 \* log_2(1/4)) +  B(1/4 \* log_2(1/4)) + ... = 2


```python
p = [0.25, 0.25, 0.25, 0.25] # ABCD的出现概率
H = entropy(p)
print '小抄的信息量：{}'.format(H)
```

    小抄的信息量：2.0


100道题的答案就是100个样本的标签。所以这段故事隐含了一个概念：**数据集中，样本标签的信息量的多少=编码数据集中所有样本的标签需要最短字符的长度**。上面的例子中，100题目的答案中有多少信息量=最少需要用几个bit才能描述清楚所有题目的答案。也就是说，**信息量是一个可以度量的编码长度，以bit为单位**。

    我在考前研究了老师的“出题特征”，发现奇数项考题的答案不是“A”就是“C”，偶数项考题的答案不是“B”就是“D”。对这一特征进行分析之后，我决定缩短小抄的编码方式为：0-A或B，1-C或D。小抄的内容如：1100……11（CDAB……CD）。小明拿到小抄，对着“题目编号是奇数还是偶数”这一特征，将小抄答案解读出来。
    
注意，当我分析出题目集的一个有效“特征”之后，考场上传递给小明的小抄的编码长度立马缩减了。特征“奇数项考题AC，偶数项考题BD”将原来的套题集划分成奇数集和偶数集两块。通过香农公式计算下整体的信息量：
H(X) = 1/2 \* H(奇数题集) + 1/2 \* H(偶数题集)

![](img/1/7/zuobi1.png)


```python
p_1 = [0.5, 0.5] # AC的出现概率
frac_1 = 50.0 / 100.0 # 奇数题集的比例

p_2 = [0.5, 0.5] #BD的出现概率
frac_2 = 50.0 / 100.0 # 偶数题集的比例

H = frac_1 * entropy(p_1) + frac_2 * entropy(p_2)
print '小抄的信息量：{}'.format(H)
```

    小抄的信息量：1.0


也就是说，**按有效的特征条件划分数据集时，数据集的标签信息量会减少。这个特征越有效，信息量减少的越多。**

    我和小明如果一起研究了出题老师的一些其他特征，比如“如果奇数题答案是A，偶数题目答案就一定是B；奇数题答案是C，偶数题答案就一定是D”，将编码长度进一步缩短为0.5（0-AB，1-CD，小抄内容为10……1-CDAB……CD），小抄的信息量进一步下降。渐渐地，当分析的特征足够多后，小明发现他已经不需要我传的小抄，就知道所有题目的答案了。

![](img/1/7/zuobi2.png)

这时，小抄的信息量为0，所有题目的答案（标签）已知。也就是说，**对于一个样本集，分类出所有样本的标签就是让样本集标签相关的信息量缩减至0**。本质上，信息是为了描述“不确定性”的一个概念，所以当所有样本标签已知，对应的信息量也就是0了。

举例传小抄只是为了有趣，做人应该诚信唯美。

## 决策树

上面的故事是一个通过特征分割样本集，缩减样本集的信息量，最终分类出所有样本的例子。决策树模型就按照这一思路设计，**决策树正是不断选择使数据集整体信息量下降最快的特征作为节点建立的树模型**。决策树和大部分机器学习模型一样，可以用于分类和回归问题上，对应的实现封装在sklearn\.tree\.DecisionTreeClassifier和sklearn\.tree\.DecisionTreeRegressor模块中。

### 对比线性模型和决策树模型的表现

下面将在“泰坦尼克号生存预测”任务上，对比逻辑分类和决策树上的表现成绩。

在同样的特征集下，我们先看线性模型基于交叉验证（Cross-validation）（请自行百度交叉验证相关知识）的准确率成绩：


```python
titanic.estimator.logistic_regression()
titanic.cross_val_accuracy_score()
```

    accuracy mean: 0.809183974577





    array([ 0.83333333,  0.81111111,  0.78651685,  0.84269663,  0.82022472,
            0.7752809 ,  0.78651685,  0.80898876,  0.80898876,  0.81818182])



接着看下非线性模型：决策树的表现成绩，sklearn封装的决策树参数“criterion”选择决策树衡量最佳分裂特征的方式，“gini”(默认，基尼杂质)或者“entropy”(信息减少量)。gini是从另一个角度评估选择最适合的特征，两者效果差异不大，这里选择“entropy”。

注意这里用grid search搜索决策树的最优层数深度参数max_depth：


```python
# 切换决策树
titanic.estimator.decision_tree_classifier(criterion='entropy')
# grid seach寻找最优的决策树层数
param_grid = dict(max_depth=range(3, 10))
best_score_, best_params_ = titanic.grid_search_common_clf(param_grid, cv=10, scoring='accuracy')
best_score_, best_params_
```




    (0.81144781144781142, {'max_depth': 3})




```python
titanic.estimator.decision_tree_classifier(criterion='entropy', **best_params_)
titanic.cross_val_accuracy_score()
```

    accuracy mean: 0.811443366247





    array([ 0.81111111,  0.81111111,  0.78651685,  0.85393258,  0.82022472,
            0.79775281,  0.79775281,  0.78651685,  0.84269663,  0.80681818])



可以看到，在titanic数据集上，相同特征条件下，非线性模型确实能比线性模型表现得更好一点。我们可以将决策树逻辑画出来，看看模型具体如何选择特征层级：


```python
# 依赖python的pydot和graphviz包
from sklearn import tree
import pydot 
from sklearn.externals.six import StringIO

# 为了方便试图，这里限制决策树的深度观察
titanic.estimator.decision_tree_classifier(criterion='entropy', max_depth=3)
clf = titanic.fit()

# 存储树plot
dotfile = StringIO()
tree.export_graphviz(clf, out_file=dotfile, feature_names=titanic.df.columns[1:])
pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")
!open dtree2.png
```

![](img/1/7/dtree2.png)

可以看到，模型先选择“是否是女性”作为第一特征，后来依次选择“是不是三等舱”等其它特征分割数据集，每层整体数据集的信息量（entropy），即每层每个数据集信息量的概率和在逐渐减少，最终构建出决策树模型。

## 集成学习

集成学习又叫做模型融合。

模型学习的本质和人相似，就是**归纳**。而归纳是需要假设前提的，比如假设新的事物的分布符合某种规律。每个模型都有自己的归纳假设，帮助模型从样本中提取主要因素，忽略次要因素，没有假设的模型也就没有了学习的能力。从这个视角看，过拟合其实就是过渡关注了样本表现出的次要因素。

我们可以将模型按假设解决问题的方式比喻为“模型的个性”。不同个性的模型运用不同的思路分析问题，比如KNN通过权衡相似的近邻归纳；线性模型假设特征的作用是线性累积的；SVM（说明：Support Vector Machine 支持向量机，读者可以自行研究，或者关注模型百科的后续系列）则是通过权衡样本分布的边界距离找到类别分界。那么，如果用KNN分析数据问题时，按相似度计算出的样本没有呈现出明显的聚集现象？或者用线形模型时，数据内在特征以非线性的方式作用呢？又或者使用SVM，但发现样本分布分散，没有集中在数据的边界呢？

模型的个性带来的问题是**模型只会以自己的擅长的模式解读样本**，他们并不像人的思维那么辩证。于是显而易见的缺陷就是，**当数据分布适应模型的个性时，成绩理想；反之，模型成绩变得糟糕。**

很多时候，面对样本分布倾向并不明确的任务，我们需要消除模型的独立个性，让模型学会用多个视角观察样本集。

### 融合成群体(Ensamble)

> 数学中有个简单有趣的理论，叫做“孔多塞陪审团定理”。大意是说：如果一个群体中每个人做出正确判断的概率高于50%，成为“大概率正确”，那么这个群体通过投票会议获得的正确率就会融合多个大概率；如果群体中的个体无限多，那么群体的正确率可以达到100%。因为群体中每个人决策判断中，错误的部分是有个体差异的，并且比例总是少数的；群体的个体之间通过投票，用多数的正确票遮盖了少数的错误票，最终群体表现可以获得高于个体表现的成绩。

![](img/1/7/voting.png)

我们可以借鉴这一想法，让**多个个性差异的模型形成模型群体**，共同发挥作用，从而获得更好的表现成绩，这一实现思路称之为“模型融合(Ensamble)”。

模型群体的好处很鲜明，群体系统有着更高的健壮性，更不容易发生过拟合，因此表现成绩更好。这一点就好比一个人做决策和一个群体做决策之间的对比，个人的主观偏见会被降低，做出的决策更通用；我们也可以理解为**模型融合就是在消除每个模型个体的“个性”，去过拟合，使整体最终表现出更好的成绩**。

简单梳理下，模型的“个性”来源是个体模型之间分析处理数据集的方式和角度的差异，所以群体对个体模型的要求是：

- 个体正确率大于50%
- 个体判断问题存在差异，有“个性”

这种尤其适合非线性模型，因为模型的表达能力非常强了，更容易过度拟合数据集，表现为成绩在不同数据集中波动程度更大，适应力更低。比如：决策树模型。

决策树可以通过构造极度丰富的特征和极深的树深度使其训练集成绩接近满分；但对应的模型适应力会最低。通常情况下的决策树靠上的特征层级晃动时，也会造成成绩的巨大差异，举个例子，在“泰坦尼克号生存预测”问题中，如果不选“是否为女性”作为顶层特征节点时的成绩差异就会非常大。简单地说，模型能力强、有个性（不稳定），这些特性自然使决策树成为理想的群体组成成员。

模型群体可以通过不同类型的模型个体组成，也可以通过使用一些技巧手段构造出的有差异的同类模型组成。具体实现起来，按思路的不同，融合方式可以分为三种：

- Bagging
- Boosting
- Stacking

其中Bagging和Boosting用在融合同类型的模型中，Stacking用在融合不同类型的模型。

### Bagging：随机森林(Random Forest)

Bagging应用于同类型的模型个体，使其形成模型群体。在Bagging中，每个模型不再训练全部的训练集样本，而是在训练集随机有放回抽样的子集中训练模型；预测样本时等权重投票。也就是说**Bagging通过构造数据集的随机子集保证模型个体差异，等权重投票融合模型个体的预测。**

基于Bagging思路改进实现的决策树群体模型就是随机森林(Random Forest)。随机森林在实现上不仅仅使用训练集的子集，同时在决策树建立树节点时只在特征集的子集中挑选。由于决策树特征层级(特征节点的选择顺序)对成绩影响很大的特性，这种方式使构造出的个体内在的“个性”很鲜明，成为理想的群体成员。

下面依旧以“泰坦尼克号生存预测”为例，观察其表现。

先看下决策树的成绩：


```python
# 决策树
titanic.estimator.decision_tree_classifier()
# grid seach寻找最优的决策树层数
param_grid = dict(max_depth=range(3, 10))
_, best_params_ = titanic.grid_search_common_clf(param_grid, cv=10, scoring='accuracy')
titanic.estimator.decision_tree_classifier(**best_params_)
titanic.cross_val_accuracy_score()
```

    accuracy mean: 0.812580013619





    array([ 0.81111111,  0.82222222,  0.7752809 ,  0.85393258,  0.82022472,
            0.7752809 ,  0.79775281,  0.78651685,  0.85393258,  0.82954545])



同样特征条件下，随机森林的表现成绩：


```python
# 随机森林
titanic.estimator.random_forest_classifier()
# grid seach寻找最优的参数：n_estimators个体模型数量；max_features特征集子集样本比例；max_depth层数深度
param_grid = {
    'n_estimators': range(80, 150, 10),
    'max_features': np.arange(.5, 1.,.1).tolist(),
    'max_depth': range(1, 10) + [None]
}
# n_jobs=-1开启多线程
best_score_, best_params_ = titanic.grid_search_common_clf(param_grid, cv=10, scoring='accuracy', n_jobs=-1)
best_score_, best_params_
```




    (0.83501683501683499,
     {'max_depth': 8, 'max_features': 0.6, 'n_estimators': 80})




```python
titanic.estimator.random_forest_classifier(**best_params_)
titanic.cross_val_accuracy_score()
```

    accuracy mean: 0.830657984338





    array([ 0.78888889,  0.77777778,  0.75280899,  0.85393258,  0.93258427,
            0.83146067,  0.82022472,  0.79775281,  0.8988764 ,  0.85227273])



比起之前的成绩，我们又有了一小步的提升，群体智慧的胜利！

### Boosting

Bagging利用随机采样集合的方式构造个体差异，一些反对者认为这种随机性虽然保证了“个性”，却显得盲目。Boosting的思路很有针对性：**依次生成了一个模型个体序列M(M_1, M_2..., M_n)，其中后续的模型尝试修正前面模型的错误。**

Boosting有很多种版本实现，最常用的是Ada-Boosting和Gradient-Boosting，可以用于分类问题和回归问题。假设数据集样本总数为m，要构造的模型群体中模型个数为n：

- 目标：获得融合后的模型M(M_1, M_2..., M_n)

Ada-Boosting的核心思想是**使序列的下一个模型更关注之前的错误样本**。而Gradient-Boosting的核心思想是：**序列的后续模型不再直接预测数据集的预测值，而是预测之前模型的预测值和真实值的差值**：

将Gradient-Boosting用于决策树模型上之后，就是GBDT(Gradient Boosting Decision Tree梯度增强决策树)。接下来看下这个模型在titanic任务上的表现成绩：


```python
# GBDT
titanic.estimator.gbdt_classifier()

# grid seach寻找最优的参数：n_estimators个体模型数量；max_depth层数深度
param_grid = {
    'n_estimators': range(80, 150, 10),
    'max_depth': range(1, 10)
}

# n_jobs=-1开启多线程, cv缩小到5，for speed
best_score_, best_params_ = titanic.grid_search_common_clf(param_grid, cv=5, scoring='accuracy', n_jobs=-1)

best_score_, best_params_
```




    (0.83501683501683499, {'max_depth': 5, 'n_estimators': 140})




```python
titanic.estimator.gbdt_classifier(**best_params_)
titanic.cross_val_accuracy_score()
```

    accuracy mean: 0.828361139485





    array([ 0.82222222,  0.8       ,  0.76404494,  0.84269663,  0.8988764 ,
            0.82022472,  0.83146067,  0.78651685,  0.85393258,  0.86363636])



经验上看，大部分分类任务中，GBDT的成绩一般和随机森林差别不大，有些任务中GBDT能够略微好一点。

在上面的例子中，gbdt_classifier函数中使用了一个sklearn.ensemble包之外的GBDT实现类库XGBoost。XGBoost是个目前非常流行的机器学习库，它对并行计算做了专门的优化，并且在xgboost.sklearn中有完全sklearn风格的封装接口。

### Stacking

Stacking就是我们常说的集成学习，用于不同类型的模型个体之间的融合。

我们接触的线性模型、决策树、随机森林、GDBT等模型，这些模型对数据集都有着不同类型的“先天偏见”。比如线性类模型总是认为数据集的内在表达式是线性的、决策树系的模型则认为内在属性可以通过树形表示——对事物的特定解读方式本身就是带着一种特定偏见。**直观角度看：不同类型的模型融合的意义在于通过群体形成的智慧弥补个体自身的先天缺陷；机器学习角度看：模型融合就是去过拟合。**

Stacking实现起来很简单，模型的先天偏见保证了成员之间的个性差异。将不同模型训练的结果以某种方式融合在一起就可以了，常用的融合方式有两种：

1. 加权投票：如将每个模型的成绩作为权重，最终预测值是每个模型乘以权重，然后相加
2. 通过一个新的模型融合多个模型个体，新模型一般是线性模型

第二种方法一般用线性模型(LR)，将每个模型对样本的预测作为特征输入，真实值作为待预测的y。用简单线性模型的原因是模型内在可挖掘的非线性因素已被模型群体充分挖掘，剩下的就是用线性模型对每个成员合理分配权重就可以了。

下面将在titanic任务上实现第二种融合方法，第一种融合方式可以参见sklearn.ensemble.VotingClassifier模块，实际中一般也是第二种方式效果更好点。

首先，选择以下几个分类器：


```python
# 逻辑分类
titanic.estimator.logistic_regression()
titanic.cross_val_accuracy_score()
```

    accuracy mean: 0.806974804222





    array([ 0.82222222,  0.81111111,  0.7752809 ,  0.83146067,  0.80898876,
            0.76404494,  0.78651685,  0.79775281,  0.83146067,  0.84090909])




```python
# 随机森林
param = {'max_depth': 8, 'max_features': 0.6, 'n_estimators': 80}
titanic.estimator.random_forest_classifier(**param)
titanic.cross_val_accuracy_score()
```

    accuracy mean: 0.828411077063





    array([ 0.8       ,  0.77777778,  0.74157303,  0.86516854,  0.91011236,
            0.84269663,  0.80898876,  0.79775281,  0.87640449,  0.86363636])




```python
# GBDT
param = {'max_depth': 5, 'n_estimators': 140}
titanic.estimator.gbdt_classifier(**param)
titanic.cross_val_accuracy_score()
```

    accuracy mean: 0.82947196686





    array([ 0.8       ,  0.82222222,  0.76404494,  0.84269663,  0.8988764 ,
            0.80898876,  0.82022472,  0.82022472,  0.86516854,  0.85227273])



将这几个模型的预测值作为x，注意到这里要融合的是模型预测的概率值而不是标签值(可以认为标签值丢失了一部分信息)，真实标签依然是y，做逻辑分类：


```python
# 准备训练好的模型
titanic.estimator.logistic_regression()
lr = titanic.fit()
param = {'max_depth': 8, 'max_features': 0.6, 'n_estimators': 80}
titanic.estimator.random_forest_classifier(**param)
rf = titanic.fit()
param = {'max_depth': 5, 'n_estimators': 140}
titanic.estimator.gbdt_classifier(**param)
gbdt = titanic.fit()

# 构造stacking训练集，融合三个模型的预测的概率值作为特征数据
x_stk = np.array([lr.predict_proba(x)[:, 0], rf.predict_proba(x)[:, 0], gbdt.predict_proba(x)[:, 0]]).T
x_df_stk = pd.DataFrame(x_stk, columns=['lr', 'rf', 'gbdt'])
y_df = pd.DataFrame(y, columns=['y'])
df = y_df.join(x_df_stk)

# stacking模型
stackings = AbuML(x_stk, y, df)

stackings.estimator.logistic_regression()

# 获得titanic的融合模型stk
stk = stackings.fit()
```

stk就是我们在已有数据集上能够得到的最好模型了。

#### 信息泄露

现在有了融合后的模型，我们希望对比下和单个模型的表现成绩，这时需要在新的数据集而不能在原有的数据做cross_validation分数测试，想想原因？


```python
stackings.cross_val_accuracy_score()
```

    accuracy mean: 0.94500453978





    array([ 0.92222222,  0.95555556,  0.91011236,  0.94382022,  0.97752809,
            0.96629213,  0.93258427,  0.93258427,  0.97752809,  0.93181818])



惊人的提升，但这个提升合理么？与之前的模型不同，stk模型的输入特征是在所有数据集上fit训练好的模型的预测值，也就是说stk的每个输入样本的数据都是在整个数据集上训练的成果，每条样本都隐含整个数据集的信息，在这上面做训练-测试集划分显然是没有意义的，不管什么样的划分方式，都达不到我们期望的训练集与测试集信息完全隔绝的效果，导致成绩虚高，这种现象叫做“信息泄露”。

解决办法如下，通过K-folder分割数据集，在封闭的训练集上完成模型的训练及融合，然后在测试集上对比成绩。


```python
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

def lr_model(x_train, x_test, y_train, y_test):
    """返回训练好的逻辑分类模型及分数"""
    lr = LogisticRegression(C=1.0)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return lr, metrics.accuracy_score(y_test, y_pred)

def rf_model(x_train, x_test, y_train, y_test):
    """返回训练好的随机森林模型及分数"""
    param_grid = {
    'n_estimators': range(80, 120, 10),
    'max_features': np.arange(.6, .9,.1).tolist(),
    'max_depth': range(3, 9) + [None]
    }
    
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    rf = RandomForestClassifier(**grid.best_params_)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    return rf, metrics.accuracy_score(y_test, y_pred)

def gbdt_model(x_train, x_test, y_train, y_test):
    """返回训练好的GBDT模型及分数"""
    param_grid = {
    'n_estimators': range(80, 120, 10),
    'max_features': np.arange(.6, .9,.1).tolist(),
    'max_depth': range(3, 9) + [None]
    }
    
    grid = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    gbdt = GradientBoostingClassifier(**grid.best_params_)
    gbdt.fit(x_train, y_train)
    y_pred = gbdt.predict(x_test)
    return gbdt, metrics.accuracy_score(y_test, y_pred)

def stack_models(x_train, x_test, y_train, y_test):
    """返回融合后的模型及分数"""
    param_grid = {
    'C': [.01, .1, 1, 10]
    }
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    stk = LogisticRegression(penalty='l1', tol=1e-6, **grid.best_params_)
    stk.fit(x_train, y_train)
    y_pred = stk.predict(x_test)
    return rf, metrics.accuracy_score(y_test, y_pred)

kf = KFold(len(titanic.y), n_splits=5, shuffle=True)
lr_scores = []
rf_scores = []
gbdt_scores = []
stk_scores = []

for train_index, test_index in kf:
    x_train, x_test = titanic.x[train_index], titanic.x[test_index]
    y_train, y_test = titanic.y[train_index], titanic.y[test_index]
    
    # 单个模型成绩
    lr, lr_score = lr_model(x_train, x_test, y_train, y_test)
    rf, rf_score = rf_model(x_train, x_test, y_train, y_test)
    gbdt, gbdt_score = gbdt_model(x_train, x_test, y_train, y_test)
    
    # stacking
    x_train_stk = np.array([lr.predict_proba(x_train)[:, 0], rf.predict_proba(x_train)[:, 0], gbdt.predict_proba(x_train)[:, 0]]).T
    x_test_stk = np.array([lr.predict_proba(x_test)[:, 0], rf.predict_proba(x_test)[:, 0], gbdt.predict_proba(x_test)[:, 0]]).T
    stk, stk_score = stack_models(x_train_stk, x_test_stk, y_train, y_test)
    
    # append score
    lr_scores.append(lr_score)
    rf_scores.append(rf_score)
    gbdt_scores.append(gbdt_score)
    stk_scores.append(stk_score)
    
print 'lr mean score: {}'.format(np.mean(lr_scores))
print 'rf mean score: {}'.format(np.mean(rf_scores))  
print 'gbdt mean score: {}'.format(np.mean(gbdt_scores))
print 'stk mean score: {}'.format(np.mean(stk_scores))
```

    lr mean score: 0.803540267403
    rf mean score: 0.809177076141
    gbdt mean score: 0.813684012303
    stk mean score: 0.801330738811


在这个数据任务中，stacking模型融合对准确率成绩似乎没有多大影响，但如果比较一下成绩在不同folder的具体波动程度：


```python
print 'lr std score: {}'.format(np.std(lr_scores))
print 'rf std score: {}'.format(np.std(rf_scores))  
print 'gbdt std score: {}'.format(np.std(gbdt_scores))
print 'stk std score: {}'.format(np.std(stk_scores))
```

    lr std score: 0.0482251333467
    rf std score: 0.0319151813243
    gbdt std score: 0.0223669765661
    stk std score: 0.0170609211563


可以看到，融合后的模型方差更小，成绩更稳定。下面展开几个模型的成绩观察下：


```python
rf_scores
```




    [0.83240223463687146,
     0.84269662921348309,
     0.797752808988764,
     0.8202247191011236,
     0.7528089887640449]




```python
gbdt_scores
```




    [0.82122905027932958,
     0.8314606741573034,
     0.8202247191011236,
     0.8258426966292135,
     0.7696629213483146]




```python
stk_scores
```




    [0.81564245810055869,
     0.8146067415730337,
     0.7865168539325843,
     0.8146067415730337,
     0.7752808988764045]



在不同的数据区间中，融合后的模型明显波动更小，说明融合后的模型对新数据集有更好的适应性。因为titanic样本数量很少，每次数据集划分都有信息损失，我们在全部数据集中完成训练模型及融合，在Kaggle的线上测试集中，确实可以看到stacking融合后的模型在未知数据集成绩有一个小幅的提升：约1%左右。

经验上看，在中小数据量级别的任务中，融合消除多个模型的缺陷，使模型整体更健壮，这一方式对成绩的提升还是比较明显的。一般会选择几个实现思路差异程度较大的模型，如：线性模型、SVM(如果训练时间允许)、随机森林、boosting模型等。由于titanic任务中数据集过小，我们将模型融合与单个模型训练放在了同一数据集中，如果数据量非常充分，更合理的做法是分为三个数据集，一个参考分割比例：

![](img/1/7/split_f.png)

## 总结

回顾整个“泰坦尼克号生存预测”任务，工程实现基本分为这么几步：

1. 数据预处理
2. 选择模型
3. GridSearch刷选模型预设参数
4. 训练模型
5. 优化：数据预处理优化、特征优化或者模型优化（更换模型 or 模型融合）

在完成一个机器学习任务中，数据集中噪声的比例决定了机器学习模型的成绩上限。对模型最终成绩的影响因素排序的话：**原始的数据质量、特征质量 >> 特征工程 > 模型工程、模型融合**。对于海量数据级别的机器学习任务，数据质量、特征质量决定了最终成绩的绝大部分，优质特征下的海量级别数据，一个简单的模型就可以比较接近模型能做到的最好成绩了。
