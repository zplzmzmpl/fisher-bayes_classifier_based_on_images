# fisher-bayes_classifier_based_on_images
realization of fisher and bayes classifiers, using c++(only achieve classification of 3 features, if you want to classify more features, just modify this code as you need.)

> [!WARNING]  
> 如果您是中国地质大学武汉的学生并且在学习《模式识别》课程，请谨慎使用。❌
> 
> 该仓库代码按老师要求创建。😈

## 1 Bayes Classifier

贝叶斯分类器是一种基于贝叶斯定理和特征条件独立假设的分类方法[^1]。
<details>
<summary>其实现原理和步骤大致如下：</summary>

 1. **准备阶段**：确定特征属性，并对每个特征属性进行适当划分，然后由`人工`对一部分待分类项进行分类，形成训练样本集合。
 2. **分类器训练阶段**：计算每个类别在训练样本中的出现`频率`及每个特征属性划分对每个类别的`条件概率估计`，并将结果记录。
 3. **应用阶段**：使用分类器对待分类项进行分类2。
 
</details>

### 1.1 输入类数,特征数,待分样本数
<div align=center><img width="600" height="400" src = "https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/90aca060-9017-4e91-a947-d1f87e918f58"></div>
<div align=center>Fig1.手动选取建立样本集</div>

### 1.2 输入训练样本数和训练样本集
由于图片中水果数量与种类较多，选取水果类（橘子与苹果）以及背景类，进行三分类。在上步骤中选定apple样本52个，橘子样本48个，背景样本53个，同时计算先验概率。
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/ca35f4c2-611f-48cf-8ca6-fdfde8a24550"></div>
<div align=center>Fig2.三样本数据集示意</div>

### 1.3 计算先验概率
根据各样本数量以及样本总数计算先验概率。
<div align =center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/7c2e194a-f4ea-4f51-86f8-a5b24f1f0707"></div>
<div align=center>Fig3.计算先验概率</div>

### 1.4 计算各类条件概率密度
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/ee756eb1-4158-4881-b02d-61c9af1b80e9"></div>
<div align=center>Fig4.创建水果结构体（RGB通道三特征）</div>

### 1.5 计算各类的后验概率
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/2f870fcd-5f30-4985-9435-de938768661c"></div>
<div align=center>Fig5.后验概率计算</div>

### 1.6 若按最小错误率原则分类,则根据后验概率判定
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/930efda0-5cb8-48dd-b25f-35e53d065dc4"></div>
<div align=center>Fig6.依据特征值分类函数</div>

### 1.7 使用测试集RGB数值数据
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/02280898-97c5-4cd9-b17c-3099c2c3b39d"></div>
<div align=center>Fig7.测试集（第四列为真值）</div>

---

*结果如下：*
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/edf629af-b3d7-46d4-b32d-7d6f98ac5dfd"></div>
<div align=center>Fig8.分类结果（0为苹果1为橘子2为背景）</div>

---

*精度验证*
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/bdc504d4-d6ee-4f27-beef-82649de59b96"></div>
<div align=center>Fig9.错误率为0.181818</div>


### 1.8 使用opencv测试整张图片所有像素
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/f3ff0e12-c58e-4b7c-b89a-4e992f5452b8"></div>
<div align=center>Fig10.获取图片内所有像素RGB值</div>

---

<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/8265d66e-75a7-4013-bb93-634334227630"></div>
<div align=center>Fig11.逐值分类</div>

---

<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/7b490003-864f-41b8-93b1-bafd4f4127fb"></div>
<div align=center>Fig12.vector<int>输出为图片</div>

---
  
### 1.9 分类结果
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/00081264-e270-4895-92f4-7998841f259e"></div>
<div align=center>Fig13.Bayes分类前后对比</div>

## 2 Fisher Classifier

Fisher线性判别准则（Fisher Linear Discriminant Criterion）是一种监督学习算法，用于数据的分类。在图像处理中，我们可以将其应用于RGB多波段的多类分类[^2]。
<details>
<summary>其实现原理和步骤大致如下：</summary>

 1. **准备阶段**：首先，我们需要一组带有`标签`的RGB图像数据作为训练集。这些图像应该包含我们想要识别的所有类别。
 2. **特征提取**：对于每个RGB图像，我们可以将其看作是一个三维的数据点，其中R、G、B通道的值对应于三个维度。我们可以使用这些值作为图像的特征。
 3. **训练阶段**：使用Fisher线性判别准则，我们可以找到一个`线性函数`，该函数可以最大化不同类别之间的距离，同时最小化同一类别内部的距离。这个函数将作为我们的分类器。
 4. **分类阶段**：对于一个新的RGB图像，我们可以将其R、G、B通道的值输入到我们的分类器中，得到一个类别标签。
 
</details>

*我们的目的是将高维的数据投影到一维直线上并在投影的值中取一个阈值进行分类，如下图所示：*
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/a5cd5fea-c352-4030-b000-081dcde5cac4"></div>

在上图，很明显左边的投影更适合分类，因为两种类别（o和x）在投影直线上能轻松地找到一个阈值将其区分开来，而右边的投影方向则不适合当前分类。
所以我们需要求解一个适合的投影方向w。在理解fisher的时候，其大致的思路如下：
1.	问题的初衷在于找到一条线将坐标点向该线上投影，将这条线的方向设为w，并用该w作为假设带入，最后解出最佳w
2.	按照我们假设的w，将样本点向该直线中投影，即wTw，求出每一类样本点在投影上的均值和方差（或者说是协方差矩阵）
3.	按照类间小，类内大的目标，设立目标函数求解w
值得注意的是，我们求得的w是最终投影的平面（在这里为一维直线）方向。

**训练数据相同**

*程序实现步骤如下*：
- 计算样本均值向量
- 计算样本离散度矩阵
- 计算样本总类内离散度矩阵
- 判别阈值
- 逐像元分类（0为苹果；1为橘子；2为背景；3为未分类）
- 根据结果赋值输出图片（0为红色；1为绿色，2为蓝色；3为白色）

**结果如下：**
<div align=center><img width="600" height="400" src="https://github.com/zplzmzmpl/fisher-bayes_classifier_based_on_images/assets/121420991/84af1831-5645-48fb-962d-29a16743989c"></div>
<div align=center>Fig14.Fisher分类前后对比</div>

> [!CAUTION]
> **免责声明**
> 
> *如果您因在课程报告中使用本仓库代码或图片而造成的一切影响，仓库所有者不承担其责任*



[^1]: 贝叶斯定理是贝叶斯分类器的理论基础，其基本形式如下：
P(A|B) = P(B|A) × P(A) / P(B)
其中，P(A|B)表示在给定B的条件下A发生的概率，P(B|A)表示在给定A的条件下B发生的概率，P(A)表示A发生的概率，P(B)表示B发生的概率。在分类问题中，我们可以将类别作为事件A，特征作为事件B。贝叶斯定理可以帮助我们计算给定特征下某个类别的概率，从而进行分类。(https://cloud.baidu.com/article/3169932)
[^2]:https://zhuanlan.zhihu.com/p/431724414
