---
layout:     post
title:      "记第二次Kaggle之旅——House Prices"
date:       2017-08-21 09:31:00
author:     "Bigzhao"
header-img: "img/header-10.png"
---

```python
%matplotlib inline
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy import stats

matplotlib.style.use('ggplot')
```


```python
def combine_data(train_data, test_data):
    combined_data = train_data.append(test_data)
    combined_data.reset_index(inplace=True)
    combined_data.drop('index', axis=1, inplace=True)
    return combined_data
```


```python
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
combined_data = combine_data(train_data.drop('SalePrice', axis=1), test_data)
```

##### 先分析缺失值


```python
null_rank = combined_data.isnull().sum().sort_values(ascending=False) / float(len(combined_data))
print '总缺失数：', len(null_rank[null_rank > 0])
null_rank.head(34)
```

    总缺失数： 34





    PoolQC          0.996574
    MiscFeature     0.964029
    Alley           0.932169
    Fence           0.804385
    FireplaceQu     0.486468
    LotFrontage     0.166495
    GarageCond      0.054471
    GarageYrBlt     0.054471
    GarageFinish    0.054471
    GarageQual      0.054471
    GarageType      0.053786
    BsmtExposure    0.028092
    BsmtCond        0.028092
    BsmtQual        0.027749
    BsmtFinType2    0.027407
    BsmtFinType1    0.027064
    MasVnrType      0.008222
    MasVnrArea      0.007879
    MSZoning        0.001370
    Utilities       0.000685
    BsmtHalfBath    0.000685
    BsmtFullBath    0.000685
    Functional      0.000685
    BsmtUnfSF       0.000343
    TotalBsmtSF     0.000343
    SaleType        0.000343
    BsmtFinSF2      0.000343
    GarageCars      0.000343
    KitchenQual     0.000343
    BsmtFinSF1      0.000343
    Electrical      0.000343
    GarageArea      0.000343
    Exterior1st     0.000343
    Exterior2nd     0.000343
    dtype: float64



##### 根据给出的含义进行分析
- PoolQC： No pool的意思 补None
- MiscFeature： 没有misc的意思 补None
- Alley: 没有Alley 补None
- Fence： No Fence None
- FireplaceQu： No Fireplace None
- GarageCond：No Garage
- GarageYrBlt： No Garage 补0
- GarageFinish: 0
- GarageQual： No Garage
- GarageType： No Garage
- BsmtExposure: No Basement
- BsmtCond: No Basement
- BsmtQual: No Basement
- BsmtFinType2: No Basement
- BsmtFinType1: No Basement
- MasVnrType:   None
- MasVnrArea: 0
- MSZoning: 众数
- LotFrontage： 上网说这个可能跟地区有关，应该用neighborhood 的众数来做：
```
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
```
- Utilities：没有用的特征，观察数据可知基本一个值，丢了

下面这些都缺一个或者两个 如果补missing的话太浪费了

- BsmtHalfBath，BsmtFullBath： 0
- Functional： 'Typ'
- TotalBsmtSF:0
- SaleType: 大众数
- BsmtFinSF2: 0
- BsmtUnfSF: 0
- GarageCars: 0
- GarageArea: 0
- KitchenQual 大众数
- BsmtFinSF1  0
- Electrical  大众数
- GarageArea  0
- Exterior1st  大众数
- Exterior2nd     同上


```python
for col in ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageQual','GarageType',
            'BsmtExposure','BsmtCond','BsmtQual', 'BsmtFinType2','BsmtFinType1','MasVnrType']:
    combined_data[col] = combined_data[col].fillna('None')
```


```python
for col in ['GarageYrBlt', 'BsmtHalfBath', 'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF',
            'BsmtFinSF2', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'GarageArea', 'MasVnrArea', 'GarageFinish']:
    combined_data[col] = combined_data[col].fillna(0)
```


```python
for col in ['MSZoning', 'SaleType', 'KitchenQual', 'Electrical', 'Exterior1st', 'Exterior2nd', 'MSSubClass']:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])
```


```python
combined_data["LotFrontage"] = combined_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
```


```python
combined_data.drop('Utilities', axis=1, inplace=True)
```


```python
combined_data.Functional = combined_data.Functional.fillna('Typ')
```

##### 检查是否补充完毕


```python
null_rank = combined_data.isnull().sum().sort_values(ascending=False) / float(len(combined_data))
print '总缺失数：', len(null_rank[null_rank > 0])
```

    总缺失数： 0



```python
SalePrice = train_data.SalePrice
```


```python
train_data = combined_data[:1459]
test_data = combined_data[1460:]
```


```python
train_data['SalePrice'] = SalePrice
```

    C:\Anaconda2\lib\site-packages\ipykernel\__main__.py:1: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':


- 网上找的 进行一元方差分析


```python
import seaborn as sns
```


```python
# 一元方差分析（类型变量）
def anova(frame, qualitative):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)  # 某特征下不同取值对应的房价组合形成二维列表
        pval = stats.f_oneway(*samples)[1]  # 一元方差分析得到 F，P，要的是 P，P越小，对方差的影响越大。
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

quantity = [attr for attr in train_data.columns if train_data.dtypes[attr] != 'object']  # 数值变量集合
quality = [attr for attr in train_data.columns if train_data.dtypes[attr] == 'object']  # 类型变量集合
a = anova(train_data,quality)
a['disparity'] = np.log(1./a['pval'].values)  # 悬殊度
fig, ax = plt.subplots(figsize=(16,8))
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
plt.show()
```


![png](https://bigzhao.github.io/img/output_19_0.png)


##### encode 函数：
- 对所有类型变量，依照各个类型变量的不同取值对应的样本集内房价的均值，按照房价均值高低
- 对此变量的当前取值确定其相对数值1,2,3,4等等，相当于对类型变量赋值使其成为连续变量。
- 此方法采用了与One-Hot编码不同的方法来处理离散数据，值得学习
- 注意：此函数会直接在原来特征下存放feature编码后的值。


```python
def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['price_mean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    # 上述 groupby()操作可以将某一feature下同一取值的数据整个到一起，结合mean()可以直接得到该特征不同取值的房价均值
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0]+1)
    ordering = ordering['order'].to_dict()
#     for attr_v, score in ordering.items():
#         # e.g. qualitative[2]: {'Grvl': 1, 'MISSING': 3, 'Pave': 2}
#         frame.loc[frame[feature] == attr_v, feature+'_E'] = score
#     frame[feature] = frame[feature].map(ordering)
    return ordering
```

##### 下面这些特征是从特征的解释和取值来看，有明显的优先级的特征


```python
cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold']
```


```python
for feature in cols:
    map_d = encode(train_data, feature)
    combined_data[feature] = combined_data[feature].map(map_d)
```


```python
combined_data.shape
```




    (2919, 80)




```python
train_data = combined_data[:1459]
test_data = combined_data[1460:]
train_data['SalePrice'] = SalePrice
```

    C:\Anaconda2\lib\site-packages\ipykernel\__main__.py:3: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      app.launch_new_instance()



```python
quantity = [attr for attr in train_data.columns if train_data.dtypes[attr] != 'object']  # 数值变量集合
```

def spearman(frame, features):
    '''
    采用“斯皮尔曼等级相关”来计算变量与房价的相关性(可查阅百科)
    此相关系数简单来说，可以对上述encoder()处理后的等级变量及其它与房价的相关性进行更好的评价（特别是对于非线性关系）
    '''
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')    
features = quantity + quality
spearman(train_data, features)

##### 从常识来说 总面积大小总是跟房价息息相关的 从上面的相关性分析来看 1stFlrSF 2ndFlrSF TotalBsmtSF 的相关性都很高 我们尝试合成一个新特征


```python
train_data['TotalSF'] = train_data['1stFlrSF'] + train_data['2ndFlrSF'] + train_data['TotalBsmtSF']
test_data['TotalSF'] = test_data['1stFlrSF'] + test_data['2ndFlrSF'] + test_data['TotalBsmtSF']
```

    C:\Anaconda2\lib\site-packages\ipykernel\__main__.py:1: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':
    C:\Anaconda2\lib\site-packages\ipykernel\__main__.py:2: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app


###### 接下来用heatmap分析相关性


```python
plt.figure(figsize=(16,12))
corr = train_data.corr()
sns.heatmap(corr, cmap ='bwr', vmin=-1, vmax=1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xae8a860>




![png](https://bigzhao.github.io/img/output_32_1.png)


###### 看看一些相关性强的特征的分布情况


```python
plt.figure(figsize=(12,4))
plt.style.use('ggplot')
ax = plt.subplot(1,2,1)
plt.scatter(x=train_data.OverallQual.values,y= train_data.SalePrice.values)
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.subplot(1,2,2)
plt.scatter(x=train_data.LotFrontage.values,y= train_data.SalePrice.values)
plt.xlabel('LotFrontage')
plt.ylabel('SalePrice')

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.scatter(x=train_data.GrLivArea.values,y= train_data.SalePrice.values)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.subplot(1,2,2)
plt.scatter(x=train_data.GarageCars.values,y= train_data.SalePrice.values)
plt.xlabel('GarageCars')
plt.ylabel('SalePrice')

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.scatter(x=train_data.TotalSF.values,y= train_data.SalePrice.values)
plt.xlabel('TotalSF')
plt.ylabel('SalePrice')

plt.subplot(1,2,2)
plt.scatter(x=train_data.ExterQual.values,y= train_data.SalePrice.values)
plt.xlabel('ExterQual')
plt.ylabel('SalePrice')
```




    <matplotlib.text.Text at 0xdfe4be0>




![png](https://bigzhao.github.io/img/output_34_1.png)



![png](https://bigzhao.github.io/img/output_34_2.png)



![png](https://bigzhao.github.io/img/output_34_3.png)


###### 找到几个离群点，找出来删掉


```python
train_data.sort_values('LotFrontage', ascending=False)[:2]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
      <th>TotalSF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1298</th>
      <td>1298</td>
      <td>1299</td>
      <td>15.0</td>
      <td>RL</td>
      <td>313.0</td>
      <td>63887</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>Bnk</td>
      <td>Corner</td>
      <td>1</td>
      <td>Edwards</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>10</td>
      <td>8</td>
      <td>2008</td>
      <td>2008</td>
      <td>Hip</td>
      <td>ClyTile</td>
      <td>Stucco</td>
      <td>Stucco</td>
      <td>Stone</td>
      <td>796.0</td>
      <td>4</td>
      <td>4</td>
      <td>PConc</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>5644.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>466.0</td>
      <td>6110.0</td>
      <td>GasA</td>
      <td>5</td>
      <td>2</td>
      <td>SBrkr</td>
      <td>4692</td>
      <td>950</td>
      <td>0</td>
      <td>5642</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>12</td>
      <td>7</td>
      <td>3</td>
      <td>5</td>
      <td>Attchd</td>
      <td>2008.0</td>
      <td>4</td>
      <td>2.0</td>
      <td>1418.0</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>214</td>
      <td>292</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>480</td>
      <td>2</td>
      <td>5</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>New</td>
      <td>Partial</td>
      <td>160000</td>
      <td>11752.0</td>
    </tr>
    <tr>
      <th>934</th>
      <td>934</td>
      <td>935</td>
      <td>12.0</td>
      <td>RL</td>
      <td>313.0</td>
      <td>27650</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>HLS</td>
      <td>Inside</td>
      <td>2</td>
      <td>NAmes</td>
      <td>PosA</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>7</td>
      <td>1960</td>
      <td>2007</td>
      <td>Flat</td>
      <td>Tar&amp;Grv</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>0.0</td>
      <td>2</td>
      <td>4</td>
      <td>CBlock</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>425.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>160.0</td>
      <td>585.0</td>
      <td>GasA</td>
      <td>5</td>
      <td>2</td>
      <td>SBrkr</td>
      <td>2069</td>
      <td>0</td>
      <td>0</td>
      <td>2069</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>9</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>Attchd</td>
      <td>1960.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>505.0</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>None</td>
      <td>0</td>
      <td>11</td>
      <td>2</td>
      <td>WD</td>
      <td>Normal</td>
      <td>242000</td>
      <td>2654.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data.sort_values('GrLivArea', ascending=False)[:2]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
      <th>TotalSF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1298</th>
      <td>1298</td>
      <td>1299</td>
      <td>15.0</td>
      <td>RL</td>
      <td>313.0</td>
      <td>63887</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>Bnk</td>
      <td>Corner</td>
      <td>1</td>
      <td>Edwards</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>10</td>
      <td>8</td>
      <td>2008</td>
      <td>2008</td>
      <td>Hip</td>
      <td>ClyTile</td>
      <td>Stucco</td>
      <td>Stucco</td>
      <td>Stone</td>
      <td>796.0</td>
      <td>4</td>
      <td>4</td>
      <td>PConc</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>5644.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>466.0</td>
      <td>6110.0</td>
      <td>GasA</td>
      <td>5</td>
      <td>2</td>
      <td>SBrkr</td>
      <td>4692</td>
      <td>950</td>
      <td>0</td>
      <td>5642</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>12</td>
      <td>7</td>
      <td>3</td>
      <td>5</td>
      <td>Attchd</td>
      <td>2008.0</td>
      <td>4</td>
      <td>2.0</td>
      <td>1418.0</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>214</td>
      <td>292</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>480</td>
      <td>2</td>
      <td>5</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>New</td>
      <td>Partial</td>
      <td>160000</td>
      <td>11752.0</td>
    </tr>
    <tr>
      <th>523</th>
      <td>523</td>
      <td>524</td>
      <td>15.0</td>
      <td>RL</td>
      <td>130.0</td>
      <td>40094</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>Bnk</td>
      <td>Inside</td>
      <td>1</td>
      <td>Edwards</td>
      <td>PosN</td>
      <td>PosN</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>10</td>
      <td>8</td>
      <td>2007</td>
      <td>2008</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>Stone</td>
      <td>762.0</td>
      <td>4</td>
      <td>4</td>
      <td>PConc</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>2260.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>878.0</td>
      <td>3138.0</td>
      <td>GasA</td>
      <td>5</td>
      <td>2</td>
      <td>SBrkr</td>
      <td>3138</td>
      <td>1538</td>
      <td>0</td>
      <td>4676</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>11</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>BuiltIn</td>
      <td>2007.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>884.0</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>208</td>
      <td>406</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>New</td>
      <td>Partial</td>
      <td>184750</td>
      <td>7814.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data.sort_values('TotalSF', ascending=False)[:2]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
      <th>TotalSF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1298</th>
      <td>1298</td>
      <td>1299</td>
      <td>15.0</td>
      <td>RL</td>
      <td>313.0</td>
      <td>63887</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>Bnk</td>
      <td>Corner</td>
      <td>1</td>
      <td>Edwards</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>10</td>
      <td>8</td>
      <td>2008</td>
      <td>2008</td>
      <td>Hip</td>
      <td>ClyTile</td>
      <td>Stucco</td>
      <td>Stucco</td>
      <td>Stone</td>
      <td>796.0</td>
      <td>4</td>
      <td>4</td>
      <td>PConc</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>5644.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>466.0</td>
      <td>6110.0</td>
      <td>GasA</td>
      <td>5</td>
      <td>2</td>
      <td>SBrkr</td>
      <td>4692</td>
      <td>950</td>
      <td>0</td>
      <td>5642</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>12</td>
      <td>7</td>
      <td>3</td>
      <td>5</td>
      <td>Attchd</td>
      <td>2008.0</td>
      <td>4</td>
      <td>2.0</td>
      <td>1418.0</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>214</td>
      <td>292</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>480</td>
      <td>2</td>
      <td>5</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>New</td>
      <td>Partial</td>
      <td>160000</td>
      <td>11752.0</td>
    </tr>
    <tr>
      <th>523</th>
      <td>523</td>
      <td>524</td>
      <td>15.0</td>
      <td>RL</td>
      <td>130.0</td>
      <td>40094</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>Bnk</td>
      <td>Inside</td>
      <td>1</td>
      <td>Edwards</td>
      <td>PosN</td>
      <td>PosN</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>10</td>
      <td>8</td>
      <td>2007</td>
      <td>2008</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>Stone</td>
      <td>762.0</td>
      <td>4</td>
      <td>4</td>
      <td>PConc</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>2260.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>878.0</td>
      <td>3138.0</td>
      <td>GasA</td>
      <td>5</td>
      <td>2</td>
      <td>SBrkr</td>
      <td>3138</td>
      <td>1538</td>
      <td>0</td>
      <td>4676</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>11</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>BuiltIn</td>
      <td>2007.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>884.0</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>208</td>
      <td>406</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>New</td>
      <td>Partial</td>
      <td>184750</td>
      <td>7814.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data.drop([523, 1298, 934], inplace=True)
```

    C:\Anaconda2\lib\site-packages\ipykernel\__main__.py:1: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':



```python
combined_data.drop([523, 1298, 934], inplace=True)
SalePrice.drop([523, 1298, 934], inplace=True)
```

##### 接下来看一下要预测的值的分布 和各个特征的分布
- 网上说这样的数据是明显正偏性，可以用对数来缓解
- 正态概率图用于检查一组数据是否服从正态分布。是实数与正态分布数据之间函数关系的散点图。如果这组实数服从正态分布正态概率图将是一条直线。


```python
fig = plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(train_data.SalePrice, fit=stats.norm)
plt.subplot(122)
res = stats.probplot(train_data.SalePrice, plot=plt)
plt.show()
```


![png](https://bigzhao.github.io/img/output_42_0.png)



```python
fig = plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(np.log(SalePrice), fit=stats.norm)
plt.subplot(122)
res = stats.probplot(np.log(SalePrice), plot=plt)
plt.show()
```


![png](https://bigzhao.github.io/img/output_43_0.png)


###### 找出偏移的特征skewed feature


```python
numeric_feats = combined_data.dtypes[combined_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = combined_data[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
```


    Skew in numerical features:






<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MiscVal</th>
      <td>21.935910</td>
    </tr>
    <tr>
      <th>PoolQC</th>
      <td>21.213939</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>17.685603</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>13.139681</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>12.082427</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>11.370087</td>
    </tr>
    <tr>
      <th>LandSlope</th>
      <td>4.994554</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>4.299698</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>4.143683</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>4.001570</td>
    </tr>
  </tbody>
</table>
</div>



###### 接下来利用 Box Cox 对那些非常偏移的特征进行纠正？


```python
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    combined_data[feat] = boxcox1p(combined_data[feat], lam)
    combined_data[feat] += 1
#all_data[skewed_features] = np.log1p(all_data[skewed_features])
```

    There are 60 skewed numerical features to Box Cox transform


###### 对剩下的那些取值只代表某一类的特征进行独热编码


```python
combined_data = pd.get_dummies(combined_data)
combined_data.shape
```




    (2916, 221)



我。。。忘记drop掉index和id了。。。


```python
combined_data.drop('index', axis=1, inplace=True)
```


```python
combined_data.drop('Id', axis=1, inplace=True)
```


```python
train_data = combined_data[:1457]
test_data = combined_data[1457:]
# train_data['SalePrice'] = np.log(SalePrice)
```


```python
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
# import lightgbm as lgb
```


```python
import os
mingw_path = 'C:\Program Files\mingw-w64\x86_64-6.3.0-posix-seh-rt_v5-rev1\mingw64\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
```

#### 直接抄网上的教程 用stacking堆积模型

- Lasso may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's Robustscaler() method on pipeline


```python
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0,
                             learning_rate=0.05, max_depth=6,
                             min_child_weight=1.5, n_estimators=5000,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2, silent=1,
                             random_state =7)
```


```python

```

##### 先定义得分函数


```python
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

def compute_score(model, X, y, scoring='mean_squared_error'):
    kford = KFold(X.shape[0], shuffle=True, random_state=123, n_folds=5)
    scores = []
    for train_index, test_index in kford:
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X[train_index], y[train_index])
        predictions = model.predict(X[test_index])
        scores.append(np.sqrt(mean_squared_error(predictions, y[test_index])))
#     print scores
    return np.mean(scores), np.std(scores)
```

##### 它们的得分


```python
SalePrice.shape
```




    (1457L,)




```python
compute_score(lasso, train_data, np.log(SalePrice))
```




    (0.11835244644570317, 0.01287478608715727)




```python
compute_score(ENet, train_data, np.log(SalePrice))
```




    (0.124188330916982, 0.0093600264793566035)




```python
compute_score(KRR, train_data, np.log(SalePrice))
```




    (0.11658986692673334, 0.0032649652645105566)




```python
compute_score(GBoost, train_data, np.log(SalePrice))
```




    (0.11306407832863619, 0.0092951027612675728)




```python
compute_score(model_xgb, train_data, np.log(SalePrice))
```




    (0.11596337876004829, 0.0031693560440681735)




```python
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
```


```python
averaged_models = AveragingModels(models = (ENet, GBoost, lasso, model_xgb))
# averaged_models.fit( train_data, np.log(SalePrice))
```


```python
print(" Averaged base models score: {} std:{} \n".format(score[0], score[1]))
```

     Averaged base models score: 0.112269933613 std:0.00805747987701




```python
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, supports, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.supports = []
        for support in supports:
            self.supports.append([i for i in range(len(support)) if support[i] == True])

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(X.shape[0], n_folds=self.n_folds, shuffle=True)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for train_index, holdout_index in kfold:
                instance = clone(clf)
                self.base_models_[i].append(instance)
                instance.fit(X[:, self.supports[i]][train_index], y[train_index])
#                 print X[X.columns[self.supports[i]]].isnull().sum()
                y_pred = instance.predict(X[:, self.supports[i]][holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X[:, self.supports[i]]) for model in self.base_models_[i]]).mean(axis=1)
            for i in range(len(self.base_models_ ))])
        return self.meta_model_.predict(meta_features)
```


```python
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, lasso),
                                                 meta_model = model_xgb)

score = compute_score(stacked_averaged_models, train_data.values, np.log(SalePrice.values))
```




    TypeErrorTraceback (most recent call last)

    <ipython-input-295-4abdfb87eb51> in <module>()
          1 stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, lasso),
    ----> 2                                                  meta_model = model_xgb)
          3
          4 score = compute_score(stacked_averaged_models, train_data.values, np.log(SalePrice.values))


    TypeError: __init__() takes at least 4 arguments (3 given)



```python
print "Stacking Averaged models score: {} ({})".format(score[0], score[1])
```

    Stacking Averaged models score: 0.110610117844 (0.00579374703457)


##### 接下来用GA来进行特征选择
- pip install genetic_selection 即可


```python
from genetic_selection import GeneticSelectionCV
```


```python
selectors = [ GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="r2",
                              n_population=100,
                              crossover_proba=0.7,
                              mutation_proba=0.2,
                              n_generations=100,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              caching=True,
                              n_jobs=-1) for estimator in [ENet, GBoost, lasso] ]
```


```python
supports = []
```


```python
for selector in selectors:
    selector.fit(train_data, np.log(SalePrice))
    supports.append(selector.support_)
```

    Selecting features with genetic algorithm.
    gen	nevals	avg                          	std                      	min                      	max                          
    0  	100   	[   0.86580992  109.62      ]	[ 0.02349226  7.45892754]	[  0.8026143  87.       ]	[   0.90306215  127.        ]
    1  	73    	[   0.88240407  110.55      ]	[ 0.01290204  6.89401915]	[  0.81785606  88.        ]	[   0.89995808  127.        ]
    2  	78    	[   0.89129099  110.3       ]	[ 0.00791364  5.98915687]	[  0.863794  97.      ]    	[   0.9076172  123.       ]  
    3  	77    	[   0.89717022  112.34      ]	[  4.92938892e-03   6.85597550e+00]	[  0.88159261  96.        ]	[   0.9076172  127.       ]  
    4  	82    	[   0.90059156  114.38      ]	[  5.64419408e-03   6.15431556e+00]	[  0.87067716  96.        ]	[   0.90962926  127.        ]
    5  	72    	[   0.90300875  115.49      ]	[  4.71769246e-03   6.67457115e+00]	[   0.87554601  101.        ]	[   0.91046496  131.        ]
    6  	86    	[   0.90557271  117.87      ]	[  3.25105242e-03   6.21716173e+00]	[   0.88741983  103.        ]	[   0.9127221  131.       ]  
    7  	84    	[   0.90663919  119.62      ]	[  5.98057365e-03   7.03957385e+00]	[  0.86090283  99.        ]  	[   0.91217723  134.        ]
    8  	78    	[   0.90830102  121.44      ]	[  4.91043287e-03   5.66977954e+00]	[   0.86891617  101.        ]	[   0.91406313  135.        ]
    9  	76    	[   0.90948683  122.72      ]	[  4.04480476e-03   5.00016000e+00]	[   0.88037027  110.        ]	[   0.91454907  135.        ]
    10 	77    	[   0.91083459  123.96      ]	[  2.44268432e-03   4.89677445e+00]	[   0.89934596  112.        ]	[   0.91454907  135.        ]
    11 	80    	[   0.91173005  124.38      ]	[  2.10992076e-03   4.98954908e+00]	[   0.90248011  106.        ]	[   0.91488137  136.        ]
    12 	72    	[   0.91243835  124.05      ]	[  1.69540676e-03   4.10457062e+00]	[   0.90295258  114.        ]	[   0.91488137  134.        ]
    13 	71    	[   0.91310188  123.81      ]	[  1.45836002e-03   4.77638985e+00]	[   0.90393445  111.        ]	[   0.91521722  133.        ]
    14 	68    	[   0.91319966  123.93      ]	[  5.07007232e-03   5.35211173e+00]	[   0.86461529  113.        ]	[   0.91547181  145.        ]
    15 	77    	[   0.91407423  124.78      ]	[  1.23202024e-03   5.61530053e+00]	[   0.9068559  113.       ]  	[   0.91548739  145.        ]
    16 	82    	[   0.9140498  123.88     ]  	[  2.61771831e-03   4.39836333e+00]	[   0.89916813  115.        ]	[   0.91567038  134.        ]
    17 	84    	[   0.91466042  124.19      ]	[  1.41025097e-03   4.92685498e+00]	[   0.90430294  112.        ]	[   0.91582887  137.        ]
    18 	72    	[   0.91470261  123.9       ]	[  1.75070348e-03   4.94267134e+00]	[   0.90383178  112.        ]	[   0.91599841  137.        ]
    19 	74    	[   0.91500556  124.65      ]	[  1.18104567e-03   4.88134203e+00]	[   0.90870434  113.        ]	[   0.91624764  137.        ]
    20 	85    	[   0.91496685  125.09      ]	[  1.96834307e-03   4.98416493e+00]	[   0.90456143  115.        ]	[   0.91643266  137.        ]
    21 	78    	[   0.91514613  125.09      ]	[  1.89573610e-03   5.05785528e+00]	[   0.90494213  110.        ]	[   0.91643266  138.        ]
    22 	68    	[   0.91539247  125.07      ]	[  1.62453584e-03   4.35489380e+00]	[   0.90682051  113.        ]	[   0.91684444  138.        ]
    23 	75    	[   0.91553608  124.1       ]	[  1.58149082e-03   4.35545635e+00]	[   0.90853276  112.        ]	[   0.91681668  135.        ]
    24 	81    	[   0.91553359  123.59      ]	[  2.01976440e-03   4.55432761e+00]	[   0.90655146  111.        ]	[   0.91692416  133.        ]
    25 	75    	[   0.91604882  122.2       ]	[  1.22199432e-03   4.64542786e+00]	[   0.90764391  110.        ]	[   0.91685303  131.        ]
    26 	78    	[   0.91584709  119.8       ]	[  1.97022425e-03   5.05371151e+00]	[   0.90625321  107.        ]	[   0.91701429  134.        ]
    27 	73    	[   0.91603635  118.98      ]	[  1.66539359e-03   4.88053276e+00]	[   0.90794307  109.        ]	[   0.9170907  130.       ]  
    28 	67    	[   0.91623374  118.07      ]	[  1.74570177e-03   4.65457839e+00]	[   0.9060448  107.       ]  	[   0.91723449  129.        ]
    29 	71    	[   0.91616056  118.29      ]	[  2.19181277e-03   4.04547896e+00]	[   0.90282742  108.        ]	[   0.91731579  128.        ]
    30 	61    	[   0.91662213  117.67      ]	[  1.29985776e-03   4.00763022e+00]	[   0.9085213  106.       ]  	[   0.91731579  127.        ]
    31 	65    	[   0.91663237  117.09      ]	[  1.43992422e-03   3.92452545e+00]	[   0.90703835  108.        ]	[   0.91732779  126.        ]
    32 	78    	[   0.91679162  115.84      ]	[  1.23898794e-03   3.62965563e+00]	[   0.9078635  105.       ]  	[   0.9174493  126.       ]  
    33 	72    	[   0.91673994  115.19      ]	[  1.93735234e-03   3.71132052e+00]	[   0.90018273  105.        ]	[   0.91751549  124.        ]
    34 	82    	[   0.91682576  113.5       ]	[  1.23592722e-03   4.10974452e+00]	[   0.91108878  103.        ]	[   0.91753944  125.        ]
    35 	78    	[   0.91694098  111.87      ]	[  1.26835718e-03   4.49367333e+00]	[   0.91091805  104.        ]	[   0.91753944  127.        ]
    36 	72    	[   0.91720506  109.97      ]	[  7.80686064e-04   3.93053431e+00]	[   0.91285672  101.        ]	[   0.91763573  123.        ]
    37 	73    	[   0.91680113  108.88      ]	[  2.52020017e-03   3.09929024e+00]	[   0.89484996  100.        ]	[   0.91768063  117.        ]
    38 	81    	[   0.9170921  107.93     ]  	[  1.39489678e-03   3.06677355e+00]	[  0.90944043  99.        ]  	[   0.91766907  115.        ]
    39 	69    	[   0.91689702  107.32      ]	[  2.42541642e-03   3.09153683e+00]	[  0.90270204  97.        ]  	[   0.91770308  116.        ]
    40 	80    	[   0.91703889  107.24      ]	[  1.85861176e-03   2.89523747e+00]	[  0.90744453  99.        ]  	[   0.91771549  114.        ]
    41 	73    	[   0.9170701  105.9      ]  	[  1.63985377e-03   3.20780299e+00]	[  0.90761346  97.        ]  	[   0.91771549  113.        ]
    42 	70    	[   0.91702146  105.38      ]	[  2.01193719e-03   2.73415435e+00]	[  0.90458412  99.        ]  	[   0.91771549  112.        ]
    43 	71    	[   0.91701623  105.39      ]	[  1.56941434e-03   3.26770562e+00]	[  0.90965345  96.        ]  	[   0.91772614  117.        ]
    44 	81    	[   0.91696123  104.62      ]	[  2.01949647e-03   3.26429165e+00]	[  0.90741519  95.        ]  	[   0.91772614  118.        ]
    45 	72    	[   0.91733852  103.3       ]	[  1.23209425e-03   2.43104916e+00]	[  0.90996813  98.        ]  	[   0.91772614  111.        ]
    46 	76    	[   0.91707251  102.58      ]	[  1.66237239e-03   2.12216870e+00]	[  0.9069658  97.       ]    	[   0.91772614  110.        ]
    47 	69    	[   0.91684367  102.28      ]	[ 0.0022735   2.22297098]          	[  0.90539669  96.        ]  	[   0.91772614  109.        ]
    48 	85    	[   0.91697565  102.01      ]	[  1.92227665e-03   2.52386608e+00]	[  0.90734996  97.        ]  	[   0.91772614  114.        ]
    49 	76    	[   0.9172236  101.14     ]  	[  1.36390195e-03   2.40840196e+00]	[  0.91068085  94.        ]  	[   0.91772614  111.        ]
    50 	83    	[   0.91729414  101.36      ]	[  1.42672732e-03   2.24285532e+00]	[  0.90710658  95.        ]  	[   0.91772614  109.        ]
    51 	75    	[   0.91677979  101.55      ]	[  2.01970059e-03   2.27321358e+00]	[  0.90746464  95.        ]  	[   0.91772614  110.        ]
    52 	79    	[   0.9168532  101.42     ]  	[ 0.00265053  2.06      ]          	[  0.90028528  93.        ]  	[   0.91772614  107.        ]
    53 	81    	[   0.91681227  101.45      ]	[ 0.00246351  2.30813778]          	[  0.90122168  97.        ]  	[   0.91772614  109.        ]
    54 	76    	[   0.91693614  101.4       ]	[  2.12853944e-03   2.22710575e+00]	[  0.90575051  97.        ]  	[   0.9178173  109.       ]  
    55 	77    	[   0.9163791  100.77     ]  	[ 0.00266878  2.54501473]          	[  0.90147394  93.        ]  	[   0.9180364  108.       ]  
    56 	75    	[   0.91695663  100.9       ]	[  2.11667106e-03   2.43104916e+00]	[  0.90668823  96.        ]  	[   0.91804664  111.        ]
    57 	78    	[   0.91709093  100.55      ]	[  1.97398535e-03   2.92361078e+00]	[  0.90513937  91.        ]  	[   0.91804664  110.        ]
    58 	74    	[   0.91685013  100.68      ]	[ 0.00313096  2.44491309]          	[  0.89336278  96.        ]  	[   0.91812077  108.        ]
    59 	64    	[   0.91690114  100.28      ]	[ 0.00309872  2.11224999]          	[  0.89587538  94.        ]  	[   0.91812077  106.        ]
    60 	74    	[  0.91750331  99.94      ]  	[  1.62226126e-03   1.84835062e+00]	[  0.90858306  96.        ]  	[   0.91812077  106.        ]
    61 	81    	[  0.91746732  99.81      ]  	[  1.57624408e-03   1.95292089e+00]	[  0.90926881  96.        ]  	[   0.91812077  107.        ]
    62 	79    	[  0.91765464  99.45      ]  	[  1.21319879e-03   2.18346056e+00]	[  0.91038855  95.        ]  	[   0.91812077  109.        ]
    63 	80    	[  0.91783891  99.16      ]  	[  8.41485210e-04   2.43195395e+00]	[  0.91328876  95.        ]  	[   0.91812456  111.        ]
    64 	75    	[  0.9173075  99.13     ]    	[  1.72795061e-03   2.04770603e+00]	[  0.90767908  94.        ]  	[   0.91812456  104.        ]
    65 	75    	[  0.91723699  99.06      ]  	[ 0.00228895  2.28831816]          	[  0.90605266  94.        ]  	[   0.9181469  107.       ]  
    66 	76    	[  0.91707985  98.46      ]  	[ 0.00297524  2.36820607]          	[  0.90190514  90.        ]  	[   0.91820607  105.        ]
    67 	69    	[  0.9174089  98.29     ]    	[  1.70471508e-03   2.34219982e+00]	[  0.90953716  92.        ]  	[   0.91821022  108.        ]
    68 	73    	[  0.91749489  97.73      ]  	[ 0.00217586  1.93832402]          	[  0.90159975  94.        ]  	[   0.91821022  106.        ]
    69 	79    	[  0.91743754  97.19      ]  	[ 0.00197904  1.77028246]          	[  0.90681941  94.        ]  	[   0.91821022  102.        ]
    70 	79    	[  0.91773693  96.42      ]  	[  1.57493000e-03   1.91405329e+00]	[  0.90544245  90.        ]  	[   0.91821022  102.        ]
    71 	69    	[  0.91766419  95.65      ]  	[  1.38336928e-03   1.92548695e+00]	[  0.91203452  89.        ]  	[   0.91821022  101.        ]
    72 	73    	[  0.91774206  95.57      ]  	[ 0.00173242  1.49167691]          	[  0.90733291  92.        ]  	[   0.91821022  100.        ]
    73 	72    	[  0.91740145  95.23      ]  	[ 0.00226264  1.92797822]          	[  0.90259871  87.        ]  	[   0.91821022  101.        ]
    74 	73    	[  0.91742912  94.56      ]  	[  1.72969037e-03   2.03627110e+00]	[  0.90961544  90.        ]  	[   0.91821022  103.        ]
    75 	74    	[  0.91737792  94.37      ]  	[ 0.00232149  1.80363522]          	[  0.90385181  89.        ]  	[   0.91821022  100.        ]
    76 	81    	[  0.91782733  93.95      ]  	[  1.14951219e-03   1.89406969e+00]	[  0.9116887  89.       ]    	[   0.91824177  101.        ]
    77 	77    	[  0.91778594  93.47      ]  	[  1.32435915e-03   1.79139610e+00]	[  0.90830404  91.        ]  	[   0.91824767  101.        ]
    78 	81    	[  0.91761158  92.64      ]  	[ 0.00171833  1.67642477]          	[  0.90816781  84.        ]  	[   0.91824767  100.        ]
    79 	68    	[  0.9171673  92.47     ]    	[ 0.00272817  1.49301708]          	[  0.90033372  89.        ]  	[  0.91821022  99.        ]  
    80 	78    	[  0.91723886  92.76      ]  	[ 0.00234297  1.62554606]          	[  0.90378706  90.        ]  	[  0.91822564  99.        ]  
    81 	78    	[  0.91777946  92.12      ]  	[  1.18723225e-03   1.38043471e+00]	[  0.91100035  87.        ]  	[   0.91822564  102.        ]
    82 	75    	[  0.91716178  92.72      ]  	[ 0.00284518  2.43753974]          	[  0.90157294  88.        ]  	[   0.91822564  106.        ]
    83 	77    	[  0.91760237  92.44      ]  	[ 0.00189983  1.73389734]          	[  0.90646646  87.        ]  	[   0.91822564  100.        ]
    84 	77    	[  0.9173153  92.33     ]    	[ 0.00225193  1.54308133]          	[  0.9060647  85.       ]    	[  0.91822564  98.        ]  
    85 	80    	[  0.91730945  92.84      ]  	[ 0.00251945  2.24819928]          	[  0.90381909  88.        ]  	[   0.91822564  102.        ]
    86 	85    	[  0.91761188  92.83      ]  	[ 0.00149557  1.49033553]          	[  0.91185719  89.        ]  	[  0.91822564  99.        ]  
    87 	69    	[  0.9173863  92.98     ]    	[ 0.0019249   1.79432439]          	[  0.90879497  86.        ]  	[   0.91822564  101.        ]
    88 	76    	[  0.91778659  93.04      ]  	[  1.15209379e-03   1.89166593e+00]	[  0.91031291  89.        ]  	[   0.9182268  101.       ]  
    89 	76    	[  0.91781143  92.84      ]  	[  1.26932407e-03   1.78168460e+00]	[  0.91125578  88.        ]  	[   0.9182268  101.       ]  
    90 	78    	[  0.9174648  93.22     ]    	[ 0.00221696  2.07643926]          	[  0.90312519  90.        ]  	[   0.9182268  101.       ]  
    91 	71    	[  0.91766002  92.81      ]  	[ 0.00177553  1.57286363]          	[  0.90628091  88.        ]  	[  0.91824805  99.        ]  
    92 	69    	[  0.91760703  92.3       ]  	[ 0.0021539  1.8627936]            	[  0.90228363  89.        ]  	[   0.91824805  100.        ]
    93 	67    	[  0.91776526  92.16      ]  	[  1.29354818e-03   2.34827596e+00]	[  0.9092828  87.       ]    	[  0.91824946  99.        ]  
    94 	77    	[  0.91765202  91.97      ]  	[  1.62313664e-03   2.13286193e+00]	[  0.90922629  86.        ]  	[   0.91825189  100.        ]
    95 	74    	[  0.9175753  91.87     ]    	[  1.77971318e-03   1.87965422e+00]	[  0.90840848  87.        ]  	[  0.91825189  99.        ]  
    96 	70    	[  0.91768353  91.84      ]  	[  1.58968316e-03   1.77042368e+00]	[  0.90880494  88.        ]  	[  0.91825316  99.        ]  
    97 	72    	[  0.9176773  91.73     ]    	[  1.59102985e-03   1.97410739e+00]	[  0.90902602  88.        ]  	[   0.91825316  101.        ]
    98 	86    	[  0.91740408  91.9       ]  	[ 0.00259158  1.72336879]          	[  0.90244203  88.        ]  	[  0.91825316  97.        ]  
    99 	74    	[  0.91777196  91.98      ]  	[  1.39948350e-03   1.97474049e+00]	[  0.90794084  88.        ]  	[   0.91825316  100.        ]
    100	74    	[  0.91749923  92.21      ]  	[ 0.00202278  1.79607906]          	[  0.90813999  85.        ]  	[  0.91825316  99.        ]  
    Selecting features with genetic algorithm.
    gen	nevals	avg                          	std                      	min                        	max                          
    0  	100   	[   0.88199225  110.48      ]	[ 0.01681776  6.67005247]	[  0.80802937  94.        ]	[   0.90655378  125.        ]
    1  	80    	[   0.8931409  110.95     ]  	[ 0.0090381   7.03757771]	[  0.8640781  98.       ]  	[   0.90853004  129.        ]
    2  	75    	[   0.90053435  110.82      ]	[  6.54879002e-03   7.08855415e+00]	[  0.86998507  96.        ]	[   0.91001779  129.        ]
    3  	71    	[   0.90384977  111.24      ]	[  4.97386617e-03   7.54866876e+00]	[  0.89000231  95.        ]	[   0.91172779  129.        ]
    4  	73    	[   0.90620746  112.12      ]	[  5.18502144e-03   7.18231161e+00]	[  0.8815695  95.       ]  	[   0.91395082  129.        ]
    5  	82    	[   0.90788267  112.85      ]	[  4.53577252e-03   6.86057578e+00]	[  0.8832205  97.       ]  	[   0.91333087  129.        ]
    6  	65    	[   0.90913557  113.82      ]	[  3.88957103e-03   6.84014620e+00]	[  0.89405569  95.        ]	[   0.91550404  129.        ]
    7  	72    	[   0.91074012  114.52      ]	[  2.42241755e-03   7.09151606e+00]	[  0.90133912  95.        ]	[   0.91596214  131.        ]
    8  	74    	[   0.91201867  117.71      ]	[  3.66237814e-03   7.77340981e+00]	[  0.89102706  97.        ]	[   0.91644555  132.        ]
    9  	83    	[   0.91297323  120.32      ]	[  2.13761421e-03   7.74322930e+00]	[   0.90383957  102.        ]	[   0.91656796  138.        ]
    10 	78    	[   0.91337805  123.16      ]	[  2.38704907e-03   6.94941724e+00]	[   0.89949028  103.        ]	[   0.91814188  136.        ]
    11 	77    	[   0.91416587  123.06      ]	[  2.16247731e-03   7.09340539e+00]	[   0.90408972  107.        ]	[   0.91905229  141.        ]
    12 	78    	[   0.91446383  122.69      ]	[  2.22512216e-03   6.38701026e+00]	[   0.90367673  102.        ]	[   0.91878197  139.        ]
    13 	77    	[   0.91487396  122.51      ]	[  3.59016764e-03   6.59165381e+00]	[   0.88480111  102.        ]	[   0.91954445  137.        ]
    14 	76    	[   0.9151997  122.45     ]  	[  2.36200422e-03   6.19092077e+00]	[   0.90363537  106.        ]	[   0.91954445  136.        ]
    15 	80    	[   0.91579262  124.02      ]	[  2.00754186e-03   6.61812662e+00]	[   0.91021255  106.        ]	[   0.92030308  141.        ]
    16 	80    	[   0.91597799  123.55      ]	[  2.66135272e-03   6.23598428e+00]	[   0.90551145  108.        ]	[   0.92030308  137.        ]
    17 	70    	[   0.91656084  123.44      ]	[  2.58580490e-03   6.54418826e+00]	[   0.90086588  110.        ]	[   0.92030308  139.        ]
    18 	76    	[   0.91691399  122.58      ]	[  2.36696311e-03   5.75183449e+00]	[   0.90699007  108.        ]	[   0.92037649  142.        ]
    19 	81    	[   0.91718727  122.75      ]	[  1.76294209e-03   5.42102389e+00]	[   0.90913857  111.        ]	[   0.92067442  137.        ]
    20 	72    	[   0.91702144  121.92      ]	[  1.88177536e-03   5.06691227e+00]	[   0.91149524  108.        ]	[   0.92089482  135.        ]
    21 	79    	[   0.91731372  122.15      ]	[  1.80584705e-03   4.60081515e+00]	[   0.90764273  111.        ]	[   0.92089482  134.        ]
    22 	77    	[   0.91737806  121.67      ]	[  2.25777331e-03   4.84573008e+00]	[   0.90485101  110.        ]	[   0.92068548  135.        ]
    23 	77    	[   0.91812921  121.44      ]	[  1.63021539e-03   5.16782353e+00]	[   0.91082617  111.        ]	[   0.92157456  131.        ]
    24 	71    	[   0.91819605  122.11      ]	[  1.99892204e-03   5.48979963e+00]	[   0.90615909  110.        ]	[   0.92157456  132.        ]
    25 	74    	[   0.91811313  121.82      ]	[  2.25909877e-03   5.09780345e+00]	[   0.9065617  109.       ]  	[   0.9220588  133.       ]  
    26 	72    	[   0.91831785  121.95      ]	[  2.52985092e-03   5.39513670e+00]	[   0.90120763  107.        ]	[   0.92123725  138.        ]
    27 	78    	[   0.91857156  121.8       ]	[  1.47626763e-03   5.37401154e+00]	[   0.91275049  107.        ]	[   0.92115538  134.        ]
    28 	81    	[   0.91880993  122.03      ]	[  1.50687979e-03   4.38509977e+00]	[   0.91288121  112.        ]	[   0.92195102  132.        ]
    29 	78    	[   0.91859017  121.64      ]	[  1.92206616e-03   5.01701106e+00]	[   0.90870419  110.        ]	[   0.92197199  133.        ]
    30 	80    	[   0.91897927  121.18      ]	[  1.42404512e-03   5.15631651e+00]	[   0.91151843  111.        ]	[   0.92162967  132.        ]
    31 	82    	[   0.91840792  121.68      ]	[  2.33111190e-03   5.04753405e+00]	[   0.9070375  109.       ]  	[   0.92162967  134.        ]
    32 	89    	[   0.91881777  121.6       ]	[  2.16634278e-03   4.68401537e+00]	[   0.90910489  108.        ]	[   0.92162967  132.        ]
    33 	75    	[   0.91865274  120.26      ]	[  2.91693085e-03   4.91654350e+00]	[   0.90570861  106.        ]	[   0.92143798  134.        ]
    34 	78    	[   0.91930217  120.4       ]	[  2.03872941e-03   5.70613705e+00]	[   0.9041704  108.       ]  	[   0.92246692  135.        ]
    35 	80    	[   0.91940217  120.63      ]	[  1.52768474e-03   5.30783383e+00]	[   0.91245111  110.        ]	[   0.92246692  131.        ]
    36 	88    	[   0.91924704  120.08      ]	[  2.11794639e-03   4.90240757e+00]	[   0.90846208  107.        ]	[   0.92197479  134.        ]
    37 	89    	[   0.91907869  120.28      ]	[  2.75138517e-03   4.84165261e+00]	[   0.90192077  104.        ]	[   0.92293555  131.        ]
    38 	78    	[   0.91959611  120.27      ]	[  2.04704933e-03   4.43588774e+00]	[   0.90704423  109.        ]	[   0.92293555  133.        ]
    39 	72    	[   0.91994777  120.23      ]	[  1.65985190e-03   4.02456209e+00]	[   0.9097486  112.       ]  	[   0.92293555  131.        ]
    40 	68    	[   0.91993054  120.39      ]	[  1.80834331e-03   4.45846386e+00]	[   0.91118847  107.        ]	[   0.92286255  133.        ]
    41 	71    	[   0.91970169  120.36      ]	[  2.31608182e-03   4.40799274e+00]	[   0.9056492  109.       ]  	[   0.92286255  130.        ]
    42 	86    	[   0.91962828  120.41      ]	[  2.31774216e-03   4.44318579e+00]	[   0.90696161  106.        ]	[   0.92357339  130.        ]
    43 	78    	[   0.91984552  120.73      ]	[  1.65647815e-03   3.85189564e+00]	[   0.9133653  112.       ]  	[   0.92233124  130.        ]
    44 	75    	[   0.92000107  120.53      ]	[  2.13764404e-03   4.53090499e+00]	[   0.90813595  109.        ]	[   0.92280329  131.        ]
    45 	77    	[   0.92019565  120.36      ]	[  1.98960822e-03   3.97874352e+00]	[   0.91194858  111.        ]	[   0.92290383  130.        ]
    46 	84    	[   0.92037773  120.07      ]	[  2.04423860e-03   4.11158120e+00]	[   0.90894846  111.        ]	[   0.92292462  130.        ]
    47 	82    	[   0.91965818  120.3       ]	[  3.22022699e-03   4.16533312e+00]	[   0.90541295  107.        ]	[   0.92298133  132.        ]
    48 	70    	[   0.92031964  120.69      ]	[  2.28491088e-03   3.96659804e+00]	[   0.90696345  107.        ]	[   0.92325667  130.        ]
    49 	72    	[   0.92056831  119.94      ]	[  2.49388614e-03   3.36695708e+00]	[   0.90467155  111.        ]	[   0.92395748  128.        ]
    50 	70    	[   0.92071687  120.29      ]	[  2.30373048e-03   3.44469157e+00]	[   0.90982837  114.        ]	[   0.92417196  128.        ]
    51 	71    	[   0.92086292  119.75      ]	[  1.88442855e-03   3.87395147e+00]	[   0.91109532  108.        ]	[   0.9234949  128.       ]  
    52 	77    	[   0.92104283  119.89      ]	[  1.86914835e-03   3.69565962e+00]	[   0.9096526  111.       ]  	[   0.92393779  131.        ]
    53 	74    	[   0.92065736  119.43      ]	[  2.02468984e-03   3.86071237e+00]	[   0.91173  111.     ]      	[   0.92363555  128.        ]
    54 	73    	[   0.92104317  119.37      ]	[  1.81647380e-03   3.73270679e+00]	[   0.91094247  109.        ]	[   0.92363555  129.        ]
    55 	84    	[   0.92099167  118.64      ]	[  2.04671526e-03   3.22031054e+00]	[   0.9119147  110.       ]  	[   0.92363555  127.        ]
    56 	80    	[   0.92059517  118.62      ]	[  2.33374679e-03   3.17105661e+00]	[   0.91086173  109.        ]	[   0.92346571  128.        ]
    57 	72    	[   0.9209575  118.59     ]  	[  2.13174076e-03   3.15941450e+00]	[   0.90818393  112.        ]	[   0.92346571  128.        ]
    58 	75    	[   0.92087591  118.15      ]	[  2.08705487e-03   2.84736720e+00]	[   0.91187996  109.        ]	[   0.92346571  128.        ]
    59 	75    	[   0.92110232  117.56      ]	[  2.37965016e-03   2.86468148e+00]	[   0.90697183  108.        ]	[   0.92408268  124.        ]
    60 	75    	[   0.92112012  117.61      ]	[  2.19234563e-03   3.40850407e+00]	[   0.90965898  110.        ]	[   0.92442472  128.        ]
    61 	78    	[   0.921337  117.36    ]    	[  1.99862850e-03   3.16075940e+00]	[   0.91059943  109.        ]	[   0.92408268  126.        ]
    62 	78    	[   0.92121636  117.19      ]	[  2.00686752e-03   2.44415630e+00]	[   0.91459269  109.        ]	[   0.92408268  123.        ]
    63 	68    	[   0.92130721  117.29      ]	[  2.44527660e-03   3.36242472e+00]	[   0.90753077  104.        ]	[   0.92408268  126.        ]
    64 	77    	[   0.92141468  118.06      ]	[  1.75612459e-03   2.77423863e+00]	[   0.91485575  109.        ]	[   0.92408268  126.        ]
    65 	75    	[   0.92141893  118.13      ]	[  2.00652408e-03   2.73735274e+00]	[   0.91117365  108.        ]	[   0.92408268  124.        ]
    66 	77    	[   0.92148213  118.25      ]	[  1.69168491e-03   3.36860505e+00]	[   0.91559898  108.        ]	[   0.92408268  128.        ]
    67 	79    	[   0.92133541  118.26      ]	[  2.52758108e-03   2.64809365e+00]	[   0.90761109  112.        ]	[   0.92408268  128.        ]
    68 	76    	[   0.92142542  117.84      ]	[  2.46500449e-03   2.51682339e+00]	[   0.91055795  110.        ]	[   0.92426522  124.        ]
    69 	86    	[   0.92119844  118.24      ]	[  2.40835465e-03   2.69859223e+00]	[   0.91072551  109.        ]	[   0.92408268  127.        ]
    70 	87    	[   0.92145942  117.98      ]	[ 0.0024303   2.37899138]          	[   0.90450185  109.        ]	[   0.92425282  125.        ]
    71 	72    	[   0.92202  118.21   ]      	[ 0.00246097  2.08947362]          	[   0.9087747  109.       ]  	[   0.92425282  126.        ]
    72 	69    	[   0.9221341  118.29     ]  	[ 0.00312114  2.16469397]          	[   0.90391208  110.        ]	[   0.92425282  126.        ]
    73 	76    	[   0.92228105  118.26      ]	[ 0.00255424  1.86343768]          	[   0.90950929  113.        ]	[   0.92425282  124.        ]
    74 	84    	[   0.92248571  118.23      ]	[ 0.00255458  1.65441833]          	[   0.91222584  113.        ]	[   0.92408268  126.        ]
    75 	72    	[   0.92295699  117.94      ]	[ 0.00255113  1.31772531]          	[   0.91135405  111.        ]	[   0.92408268  123.        ]
    76 	73    	[   0.92276859  117.9       ]	[ 0.00337925  0.92195445]          	[   0.90533434  113.        ]	[   0.92408268  120.        ]
    77 	68    	[   0.92278466  117.87      ]	[ 0.00297429  1.38314858]          	[   0.90870898  111.        ]	[   0.92408268  123.        ]
    78 	80    	[   0.92295241  117.94      ]	[ 0.00219253  1.16464587]          	[   0.9122866  113.       ]  	[   0.92408268  121.        ]
    79 	79    	[   0.92270412  117.95      ]	[ 0.00487891  1.57082781]          	[   0.88398422  112.        ]	[   0.92408268  125.        ]
    80 	72    	[   0.92280523  117.92      ]	[ 0.00417141  1.41901374]          	[   0.89129277  111.        ]	[   0.92408268  124.        ]
    81 	69    	[   0.92345245  117.89      ]	[ 0.00191723  0.99894945]          	[   0.91089362  114.        ]	[   0.92408268  123.        ]
    82 	83    	[   0.92274687  117.67      ]	[ 0.00285598  1.23332883]          	[   0.90993149  113.        ]	[   0.92408268  121.        ]
    83 	71    	[   0.92268815  117.69      ]	[ 0.00316327  1.59809261]          	[   0.90911692  110.        ]	[   0.92408268  123.        ]
    84 	67    	[   0.92315278  117.7       ]	[ 0.00202347  1.65227116]          	[   0.91553011  107.        ]	[   0.92408268  122.        ]
    85 	70    	[   0.92299701  117.97      ]	[ 0.00372051  0.75438717]          	[   0.89700843  113.        ]	[   0.92408268  121.        ]
    86 	77    	[   0.92335089  118.02      ]	[ 0.00167273  1.43513066]          	[   0.91616581  111.        ]	[   0.92408268  124.        ]
    87 	87    	[   0.92259201  117.66      ]	[ 0.00362403  1.59511755]          	[   0.90652321  112.        ]	[   0.92408268  122.        ]
    88 	74    	[   0.92305483  117.75      ]	[ 0.00231461  1.42390309]          	[   0.91183005  112.        ]	[   0.92408268  124.        ]
    89 	78    	[   0.92287554  117.81      ]	[ 0.00323966  1.51456264]          	[   0.89855686  109.        ]	[   0.92408268  123.        ]
    90 	80    	[   0.92337974  117.82      ]	[ 0.00182273  1.28358872]          	[   0.91399788  112.        ]	[   0.92408268  123.        ]
    91 	67    	[   0.92331442  117.68      ]	[ 0.00198146  1.33326666]          	[   0.9148631  111.       ]  	[   0.92408268  120.        ]
    92 	74    	[   0.92212236  117.39      ]	[ 0.00370174  2.21311997]          	[   0.90813582  109.        ]	[   0.92408268  125.        ]
    93 	78    	[   0.92218849  117.79      ]	[ 0.00370329  1.58300347]          	[   0.90434825  108.        ]	[   0.92408268  123.        ]
    94 	70    	[   0.92307236  117.78      ]	[ 0.00272226  1.43234074]          	[   0.90709667  110.        ]	[   0.92408268  121.        ]
    95 	74    	[   0.92317178  117.82      ]	[ 0.00279944  1.40982268]          	[   0.90783107  110.        ]	[   0.92408268  122.        ]
    96 	75    	[   0.92261679  117.69      ]	[ 0.00301413  1.85307852]          	[   0.90908678  111.        ]	[   0.92408268  126.        ]
    97 	76    	[   0.92306876  118.03      ]	[ 0.00212656  1.33757243]          	[   0.9152614  110.       ]  	[   0.92408268  124.        ]
    98 	86    	[   0.92267316  117.9       ]	[ 0.00336692  2.11423745]          	[   0.90749821  108.        ]	[   0.92408268  126.        ]
    99 	79    	[   0.92306738  118.09      ]	[ 0.00249523  1.67985118]          	[   0.91188696  112.        ]	[   0.92408268  126.        ]
    100	81    	[   0.92255969  117.61      ]	[ 0.00381833  2.01938109]          	[   0.89651056  109.        ]	[   0.92408268  123.        ]
    Selecting features with genetic algorithm.
    gen	nevals	avg                          	std                      	min                        	max                          
    0  	100   	[   0.86793083  109.78      ]	[ 0.02486248  7.61784746]	[  0.77642611  89.        ]	[   0.90434772  129.        ]
    1  	71    	[   0.88284337  110.69      ]	[ 0.01347896  7.28243778]	[  0.82732078  92.        ]	[   0.90434772  130.        ]
    2  	69    	[   0.89150862  113.14      ]	[ 0.00895802  7.82562458]	[  0.86420148  93.        ]	[   0.90794861  130.        ]
    3  	80    	[   0.89628665  113.83      ]	[  7.71993524e-03   8.03250272e+00]	[  0.85548364  98.        ]	[   0.90820121  131.        ]
    4  	82    	[   0.90172265  117.11      ]	[  5.81791675e-03   7.28545812e+00]	[   0.87057111  101.        ]	[   0.9106758  131.       ]  
    5  	80    	[   0.90531764  119.8       ]	[  3.93422366e-03   7.25809892e+00]	[   0.88978145  104.        ]	[   0.91269345  133.        ]
    6  	76    	[   0.90735725  121.        ]	[  3.04487293e-03   7.36749618e+00]	[   0.89880249  101.        ]	[   0.91269345  138.        ]
    7  	70    	[   0.90827647  121.71      ]	[  4.17822825e-03   6.10785560e+00]	[   0.88585064  107.        ]	[   0.91300118  138.        ]
    8  	71    	[   0.90989504  121.55      ]	[  2.33632265e-03   6.34093842e+00]	[   0.89794555  105.        ]	[   0.91307995  138.        ]
    9  	73    	[   0.91098972  121.37      ]	[  1.66440733e-03   6.00442337e+00]	[   0.90532649  105.        ]	[   0.91410075  134.        ]
    10 	79    	[   0.9113377  121.16     ]  	[  2.42830775e-03   6.11182460e+00]	[   0.9003935  101.       ]  	[   0.91499101  131.        ]
    11 	69    	[   0.91220954  123.        ]	[  2.26132497e-03   5.15363949e+00]	[   0.9000761  107.       ]  	[   0.91440229  135.        ]
    12 	83    	[   0.91289845  123.95      ]	[  2.15980164e-03   5.40439636e+00]	[   0.9015498  110.       ]  	[   0.91557305  138.        ]
    13 	73    	[   0.9134155  124.66     ]  	[  1.85417947e-03   5.33332917e+00]	[   0.90159051  112.        ]	[   0.91557305  137.        ]
    14 	83    	[   0.91365536  125.13      ]	[  2.69466865e-03   5.16266404e+00]	[   0.89187922  110.        ]	[   0.91565968  138.        ]
    15 	78    	[   0.91410205  124.1       ]	[  1.70408277e-03   5.51996377e+00]	[   0.90617969  106.        ]	[   0.91634664  134.        ]
    16 	74    	[   0.91429417  123.66      ]	[  1.94494129e-03   5.22918732e+00]	[   0.90690973  106.        ]	[   0.91598771  136.        ]
    17 	72    	[   0.91442566  123.12      ]	[  3.05984463e-03   5.21589877e+00]	[   0.88963439  106.        ]	[   0.91596523  136.        ]
    18 	87    	[   0.91436196  122.95      ]	[  2.36140792e-03   4.92823498e+00]	[   0.90278605  111.        ]	[   0.91633587  133.        ]
    19 	80    	[   0.91484899  123.49      ]	[  2.69045030e-03   4.73390959e+00]	[   0.89191223  113.        ]	[   0.91638233  137.        ]
    20 	77    	[   0.91496684  123.24      ]	[  2.70617864e-03   5.14610532e+00]	[   0.89532606  108.        ]	[   0.91647122  136.        ]
    21 	77    	[   0.91560868  123.62      ]	[  1.65852400e-03   4.89035786e+00]	[   0.90103254  111.        ]	[   0.91676681  135.        ]
    22 	82    	[   0.91568302  123.48      ]	[  2.32848679e-03   5.70522567e+00]	[   0.89504071  105.        ]	[   0.91672057  135.        ]
    23 	80    	[   0.91558767  123.78      ]	[  2.13963271e-03   5.07066071e+00]	[   0.90400251  112.        ]	[   0.91673784  139.        ]
    24 	79    	[   0.91601341  122.73      ]	[  1.37013283e-03   4.62353761e+00]	[   0.90776291  109.        ]	[   0.91687568  135.        ]
    25 	79    	[   0.915727  122.32    ]    	[  2.27766145e-03   4.28224240e+00]	[   0.90418081  110.        ]	[   0.91699512  133.        ]
    26 	73    	[   0.91602097  121.42      ]	[  1.96333786e-03   4.58733038e+00]	[   0.90696503  110.        ]	[   0.91709012  133.        ]
    27 	74    	[   0.9162925  121.23     ]  	[  1.71294769e-03   4.32401434e+00]	[   0.90631372  110.        ]	[   0.91715152  133.        ]
    28 	78    	[   0.91603852  121.28      ]	[  2.51030663e-03   4.36366818e+00]	[   0.90228409  110.        ]	[   0.91735398  130.        ]
    29 	74    	[   0.91642898  120.98      ]	[  1.91175666e-03   4.63676611e+00]	[   0.9082779  113.       ]  	[   0.91735398  130.        ]
    30 	78    	[   0.91654177  119.45      ]	[  1.92966551e-03   4.35057467e+00]	[   0.90613872  112.        ]	[   0.91735398  130.        ]
    31 	80    	[   0.9165594  118.06     ]  	[  2.07813910e-03   3.83619603e+00]	[   0.90348192  105.        ]	[   0.91735398  128.        ]
    32 	79    	[   0.91665164  117.8       ]	[  1.80047557e-03   4.33589668e+00]	[   0.90683662  109.        ]	[   0.91741581  130.        ]
    33 	80    	[   0.91640096  116.98      ]	[  2.50516655e-03   4.45416659e+00]	[   0.90271752  106.        ]	[   0.9174245  131.       ]  
    34 	75    	[   0.91653202  116.29      ]	[  1.81074153e-03   3.75311870e+00]	[   0.90788322  109.        ]	[   0.91746463  125.        ]
    35 	83    	[   0.91660042  115.7       ]	[  2.26721983e-03   2.92745623e+00]	[   0.90599306  109.        ]	[   0.91754056  123.        ]
    36 	82    	[   0.91678319  115.61      ]	[  1.78268081e-03   3.51822398e+00]	[   0.90779212  106.        ]	[   0.91755693  126.        ]
    37 	72    	[   0.91674239  115.2       ]	[  2.17832809e-03   3.77624152e+00]	[   0.90575853  104.        ]	[   0.91757917  123.        ]
    38 	60    	[   0.91694656  115.14      ]	[  1.78751588e-03   3.98753056e+00]	[   0.90673273  100.        ]	[   0.91759555  121.        ]
    39 	77    	[   0.91690197  114.42      ]	[  1.81652666e-03   3.96025252e+00]	[   0.90775348  103.        ]	[   0.91759529  121.        ]
    40 	71    	[   0.9170607  114.68     ]  	[  1.50756401e-03   3.79441695e+00]	[   0.90831849  103.        ]	[   0.91760383  122.        ]
    41 	84    	[   0.91663877  114.02      ]	[  2.92505128e-03   3.06913669e+00]	[   0.89469539  106.        ]	[   0.91761964  121.        ]
    42 	79    	[   0.91680815  113.51      ]	[  2.58766132e-03   3.26648129e+00]	[   0.90185024  106.        ]	[   0.91763763  124.        ]
    43 	79    	[   0.91650971  112.85      ]	[ 0.00328909  3.15396576]          	[   0.89430932  104.        ]	[   0.91763882  121.        ]
    44 	87    	[   0.9166858  112.17     ]  	[  2.40617747e-03   3.15928789e+00]	[   0.90758093  101.        ]	[   0.91763988  121.        ]
    45 	72    	[   0.91696284  111.84      ]	[  2.05698078e-03   2.83450172e+00]	[  0.90388177  97.        ]  	[   0.91765828  118.        ]
    46 	83    	[   0.91694736  111.81      ]	[  1.96116616e-03   2.67467755e+00]	[   0.9063325  105.       ]  	[   0.917671  119.      ]    
    47 	88    	[   0.91671103  111.51      ]	[  2.05149578e-03   3.05448850e+00]	[   0.90628456  101.        ]	[   0.917671  117.      ]    
    48 	77    	[   0.91650779  112.18      ]	[  2.82850336e-03   2.97449155e+00]	[   0.90120636  102.        ]	[   0.91767369  121.        ]
    49 	74    	[   0.91707508  112.28      ]	[  1.64483727e-03   3.04985246e+00]	[   0.90782976  104.        ]	[   0.9176737  120.       ]  
    50 	72    	[   0.91717627  111.89      ]	[  1.80900660e-03   2.77811087e+00]	[   0.90410445  107.        ]	[   0.91767533  121.        ]
    51 	74    	[   0.91705212  110.97      ]	[  2.34712916e-03   2.56302556e+00]	[   0.9005983  103.       ]  	[   0.91767533  116.        ]
    52 	76    	[   0.91657582  110.82      ]	[ 0.00270621  2.49551598]          	[   0.90487975  105.        ]	[   0.91767533  116.        ]
    53 	81    	[   0.9167099  109.88     ]  	[  2.53819148e-03   2.79385039e+00]	[   0.9025173  102.       ]  	[   0.91769473  117.        ]
    54 	61    	[   0.91707451  110.39      ]	[  2.01106704e-03   2.15357842e+00]	[   0.9040839  105.       ]  	[   0.91774695  117.        ]
    55 	62    	[   0.91696331  110.71      ]	[  1.92519398e-03   2.24630808e+00]	[   0.90827923  102.        ]	[   0.9178148  117.       ]  
    56 	76    	[   0.91682756  110.88      ]	[ 0.00254247  2.41776757]          	[   0.90280253  104.        ]	[   0.91775138  118.        ]
    57 	67    	[   0.91621653  110.52      ]	[ 0.00417124  2.05173098]          	[   0.88825013  104.        ]	[   0.9178148  115.       ]  
    58 	80    	[   0.9164481  110.43     ]  	[ 0.00334143  2.70279485]          	[   0.89926589  105.        ]	[   0.9178148  120.       ]  
    59 	80    	[   0.91704419  109.59      ]	[  2.45743241e-03   2.63474856e+00]	[   0.89895659  102.        ]	[   0.9178148  116.       ]  
    60 	70    	[   0.91689827  109.7       ]	[  2.21677074e-03   2.44744765e+00]	[   0.90691431  103.        ]	[   0.9178148  118.       ]  
    61 	70    	[   0.9172063  109.05     ]  	[  1.48840962e-03   2.33398800e+00]	[   0.90853373  100.        ]	[   0.9178148  114.       ]  
    62 	76    	[   0.91723778  108.73      ]	[  1.61411257e-03   1.86469837e+00]	[   0.90700216  101.        ]	[   0.9178148  112.       ]  
    63 	81    	[   0.91704729  108.6       ]	[ 0.00260135  2.05912603]          	[   0.89717305  103.        ]	[   0.9178148  116.       ]  
    64 	78    	[   0.91734065  108.83      ]	[ 0.0016452   1.56878934]          	[   0.9054299  104.       ]  	[   0.9178148  115.       ]  
    65 	71    	[   0.91707133  108.73      ]	[ 0.0024707   1.75985795]          	[   0.90256467  104.        ]	[   0.9178148  116.       ]  
    66 	77    	[   0.91692243  108.45      ]	[ 0.00309698  2.02175666]          	[   0.89405181  102.        ]	[   0.91781481  117.        ]
    67 	75    	[   0.91673086  108.81      ]	[ 0.00308439  2.23470356]          	[   0.89327047  102.        ]	[   0.91786819  116.        ]
    68 	76    	[   0.91752812  108.72      ]	[  1.28581712e-03   1.71510933e+00]	[   0.90714649  105.        ]	[   0.91786819  117.        ]
    69 	84    	[   0.91720352  108.35      ]	[ 0.00206652  1.87283208]          	[   0.90335531  102.        ]	[   0.91786819  115.        ]
    70 	74    	[   0.91704006  108.        ]	[ 0.00264739  1.6       ]          	[  0.90240214  98.        ]  	[   0.91786819  111.        ]
    71 	83    	[   0.91670018  107.82      ]	[ 0.00235881  2.14653209]          	[   0.90649837  100.        ]	[   0.91786819  115.        ]
    72 	74    	[   0.91732443  107.56      ]	[  1.53320775e-03   1.58946532e+00]	[   0.90919842  103.        ]	[   0.91786819  113.        ]
    73 	79    	[   0.91713315  107.41      ]	[ 0.00207153  1.289147  ]          	[   0.90680314  103.        ]	[   0.91786819  110.        ]
    74 	67    	[   0.91683697  107.32      ]	[ 0.00368572  2.15814735]          	[   0.89728292  102.        ]	[   0.91786819  117.        ]
    75 	76    	[   0.91700147  107.05      ]	[ 0.00292024  1.57082781]          	[   0.89642977  103.        ]	[   0.91786819  115.        ]
    76 	74    	[   0.91726435  107.08      ]	[  1.87901395e-03   1.91143925e+00]	[   0.90475958  104.        ]	[   0.91786819  116.        ]
    77 	83    	[   0.91692293  106.53      ]	[ 0.00282982  1.82458214]          	[  0.89987416  99.        ]  	[   0.91787242  113.        ]
    78 	73    	[   0.91720043  106.26      ]	[ 0.0024912   1.38289551]          	[   0.90121492  102.        ]	[   0.91789132  111.        ]
    79 	77    	[   0.9165999  105.79     ]  	[ 0.00433634  1.73375315]          	[   0.88772084  102.        ]	[   0.91789676  113.        ]
    80 	79    	[   0.91679085  105.55      ]	[ 0.00335779  1.6332483 ]          	[  0.89568906  99.        ]  	[   0.91789677  110.        ]
    81 	80    	[   0.91723928  105.8       ]	[  1.84430295e-03   1.91311265e+00]	[   0.90613909  102.        ]	[   0.91789676  115.        ]
    82 	73    	[   0.9169772  105.45     ]  	[ 0.00233257  2.05608852]          	[  0.90270027  96.        ]  	[   0.91789676  112.        ]
    83 	74    	[   0.91727977  106.25      ]	[ 0.00174107  1.53866826]          	[   0.90602395  100.        ]	[   0.91789677  113.        ]
    84 	75    	[   0.91746702  106.12      ]	[ 0.00177677  1.6079801 ]          	[   0.9025597  100.       ]  	[   0.91789677  114.        ]
    85 	75    	[   0.91742048  106.29      ]	[ 0.00234092  1.70467006]          	[   0.89516425  102.        ]	[   0.91789677  112.        ]
    86 	75    	[   0.91733093  106.        ]	[ 0.00210229  1.67332005]          	[   0.90223905  102.        ]	[   0.91789677  111.        ]
    87 	72    	[   0.91635768  106.08      ]	[ 0.00407595  2.48869444]          	[  0.88901233  97.        ]  	[   0.91789677  116.        ]
    88 	83    	[   0.91714746  106.49      ]	[  1.74214074e-03   2.41451859e+00]	[   0.907959  100.      ]    	[   0.91789677  119.        ]
    89 	83    	[   0.91622977  106.77      ]	[ 0.00442435  2.34032049]          	[  0.88414029  98.        ]  	[   0.91789677  114.        ]
    90 	76    	[   0.91727831  107.07      ]	[  2.04463689e-03   2.16450918e+00]	[   0.90253257  100.        ]	[   0.91789677  115.        ]
    91 	80    	[   0.9171542  107.15     ]  	[ 0.0027992   1.84051623]          	[   0.89433599  100.        ]	[   0.91789677  114.        ]
    92 	75    	[   0.91704319  107.12      ]	[ 0.00237374  2.0653329 ]          	[   0.9063189  102.       ]  	[   0.91789677  115.        ]
    93 	77    	[   0.9166723  106.92     ]  	[ 0.00376204  1.93742097]          	[   0.89031767  101.        ]	[   0.91789677  114.        ]
    94 	70    	[   0.91719034  106.33      ]	[ 0.00246183  1.40751554]          	[   0.89825724  101.        ]	[   0.91789677  111.        ]
    95 	64    	[   0.9172909  106.02     ]  	[ 0.00154588  1.3113352 ]          	[   0.91013196  102.        ]	[   0.91789677  113.        ]
    96 	75    	[   0.91722307  106.03      ]	[ 0.00263197  1.32253544]          	[  0.90092471  98.        ]  	[   0.91789677  113.        ]
    97 	68    	[   0.91656039  106.2       ]	[ 0.00435857  1.8       ]          	[  0.88694937  99.        ]  	[   0.91789677  112.        ]
    98 	71    	[   0.91622342  105.89      ]	[ 0.00493727  1.70818617]          	[   0.88479299  100.        ]	[   0.91789677  114.        ]
    99 	75    	[   0.91661709  106.03      ]	[ 0.0040995   1.62144997]          	[   0.88478154  100.        ]	[   0.91789677  114.        ]
    100	80    	[   0.91747666  106.05      ]	[  1.30288156e-03   1.49248116e+00]	[   0.91058687  101.        ]	[   0.91789677  116.        ]


- 耗时两天。。。。一方面是我设的代数有点多，发现模型基本再100代左右就已经收敛了，所以剩下的代数都是无用功
- 另外一个原因是gboost和xgmboost本来就比较慢。。


```python
compute_score(ENet, train_data[train_data.columns[supports[0]]].values, np.log(SalePrice.values))
```




    (0.11550120899152605, 0.0078702590924465115)




```python
compute_score(GBoost, train_data[train_data.columns[supports[1]]].values, np.log(SalePrice.values))
```




    (0.1150735441405851, 0.0059863590498971343)




```python
compute_score(lasso, train_data[train_data.columns[supports[2]]].values, np.log(SalePrice.values))
```




    (0.11576834449263107, 0.0078684961517621697)




```python
stacked_averaged_models_GA = StackingAveragedModels(base_models = (ENet, GBoost, lasso),
                                                 meta_model = model_xgb, supports = supports)

compute_score(stacked_averaged_models_GA, train_data.values, np.log(SalePrice.values))
```




    (0.11332569088538043, 0.009948747794297292)




```python
stacked_averaged_models_GA.fit(train_data.values, np.log(SalePrice.values))
```




    StackingAveragedModels(base_models=(Pipeline(steps=[('robustscaler', RobustScaler(copy=True, with_centering=True, with_scaling=True)), ('elasticnet', ElasticNet(alpha=0.0005, copy_X=True, fit_intercept=True, l1_ratio=0.9,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=3, selecti...ve=False, precompute=False, random_state=1,
       selection='cyclic', tol=0.0001, warm_start=False))])),
                meta_model=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=0.2, gamma=0.0, learning_rate=0.05,
           max_delta_step=0, max_depth=6, min_child_weight=1.5, missing=None,
           n_estimators=5000, n_jobs=1, nthread=None, objective='reg:linear',
           random_state=7, reg_alpha=0.9, reg_lambda=0.6, scale_pos_weight=1,
           seed=None, silent=1, subsample=0.2),
                n_folds=5,
                supports=[[0, 2, 7, 8, 9, 10, 11, 12, 16, 18, 19, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 40, 41, 42, 43, 44, 47, 48, 51, 58, 62, 66, 67, 68, 70, 71, 75, 76, 78, 79, 85, 87, 88, 93, 94, 95, 96, 97, 98, 99, 100, 108, 110, 114, 117, 118, 120, 123, 130, 131, 134, 135, 143, 148, 150,...166, 167, 169, 171, 174, 178, 180, 182, 183, 186, 189, 191, 194, 198, 201, 205, 208, 210, 213, 214]])




```python
res = np.exp(stacked_averaged_models_GA.predict(test_data.values))
```


```python
Id = pd.read_csv('test.csv').Id
```


```python
pd.DataFrame({'Id': Id, 'SalePrice': res}).to_csv('2017-8-17-11-17.csv', index=False)
```

##### 这是根据需求改的另外一个类，多了supports变量，也就是特征选择后的特征


```python
class DifferentFeatureAveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, supports):
        self.models = models
        self.supports = []
        for support in supports:
            self.supports.append([i for i in range(len(support)) if support[i] == True])

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for i in range(len(self.models_)):
            self.models_[i].fit(X[:, self.supports[i]], y)
        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            self.models[i].predict(X[:, self.supports[i]]) for i in range(len(self.models_))
        ])
        return np.mean(predictions, axis=1)
```


```python
averaged_models = DifferentFeatureAveragingModels(models = (ENet, GBoost, lasso), supports=supports)
```


```python
compute_score(averaged_models, train_data.values, np.log(SalePrice.values))
```




    (0.09155876812889685, 0.0098478977861897118)




```python
averaged_models.fit(train_data.values, np.log(SalePrice.values))
```




    DifferentFeatureAveragingModels(models=(Pipeline(steps=[('robustscaler', RobustScaler(copy=True, with_centering=True, with_scaling=True)), ('elasticnet', ElasticNet(alpha=0.0005, copy_X=True, fit_intercept=True, l1_ratio=0.9,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=3, selection='c...ve=False, precompute=False, random_state=1,
       selection='cyclic', tol=0.0001, warm_start=False))])),
                    supports=[[0, 2, 7, 8, 9, 10, 11, 12, 16, 18, 19, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 40, 41, 42, 43, 44, 47, 48, 51, 58, 62, 66, 67, 68, 70, 71, 75, 76, 78, 79, 85, 87, 88, 93, 94, 95, 96, 97, 98, 99, 100, 108, 110, 114, 117, 118, 120, 123, 130, 131, 134, 135, 143, 148, 150,...166, 167, 169, 171, 174, 178, 180, 182, 183, 186, 189, 191, 194, 198, 201, 205, 208, 210, 213, 214]])




```python
res = np.exp(averaged_models.predict(test_data.values))
```


```python
pd.DataFrame({'Id': Id, 'SalePrice': res}).to_csv('2017-8-17-15-46.csv', index=False)
```


```python
selectors_ = [ GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="mean_squared_error",
                              n_population=100,
                              crossover_proba=0.7,
                              mutation_proba=0.2,
                              n_generations=300,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              caching=True,
                              n_jobs=-1) for estimator in [ENet, GBoost, lasso, model_xgb, KRR] ]
```

##### 发现三个模型集合缺失有提高，干脆接下来集成学习五个模型


```python
supports_ = []
for selector in selectors_:
    selector.fit(train_data.values, np.log(SalePrice.values))
    supports_.append(selector.support_)
```

    Selecting features with genetic algorithm.
    gen	nevals	avg                                	std                                	min                                	max                                
    0  	100   	[ -2.06053874e-02   1.09190000e+02]	[  3.18558949e-03   8.67490058e+00]	[ -3.06123648e-02   9.10000000e+01]	[ -1.59791901e-02   1.29000000e+02]
    1  	79    	[ -1.86868118e-02   1.12020000e+02]	[  2.09376712e-03   7.15678699e+00]	[ -2.95285532e-02   9.60000000e+01]	[ -1.48742201e-02   1.26000000e+02]
    2  	74    	[ -1.76052232e-02   1.13930000e+02]	[  1.42359580e-03   6.81066076e+00]	[ -2.67606198e-02   1.01000000e+02]	[ -1.53454403e-02   1.27000000e+02]
    3  	72    	[ -1.68865880e-02   1.13710000e+02]	[  1.21273106e-03   6.08981937e+00]	[ -2.11601788e-02   1.00000000e+02]	[ -1.45947791e-02   1.29000000e+02]
    4  	86    	[ -1.60466706e-02   1.15170000e+02]	[  8.58440876e-04   6.64086591e+00]	[ -1.91112991e-02   9.90000000e+01]	[ -1.45098830e-02   1.32000000e+02]
    5  	74    	[ -1.56068114e-02   1.16220000e+02]	[  9.40952409e-04   6.83751417e+00]	[ -2.17093063e-02   1.00000000e+02]	[ -1.44154832e-02   1.35000000e+02]
    6  	71    	[ -1.51926793e-02   1.17090000e+02]	[  8.59151501e-04   5.88743577e+00]	[ -2.22630454e-02   1.03000000e+02]	[ -1.41810744e-02   1.31000000e+02]
    7  	75    	[ -1.48220388e-02   1.18650000e+02]	[  3.78541532e-04   6.38337685e+00]	[ -1.61633536e-02   1.02000000e+02]	[ -1.40988117e-02   1.31000000e+02]
    8  	75    	[ -1.45928993e-02   1.18820000e+02]	[  3.56281395e-04   5.91334085e+00]	[ -1.63837504e-02   1.03000000e+02]	[ -1.40624692e-02   1.34000000e+02]
    9  	73    	[ -1.44083407e-02   1.19110000e+02]	[  4.67649169e-04   5.92940975e+00]	[ -1.74211465e-02   1.03000000e+02]	[ -1.37337908e-02   1.37000000e+02]
    10 	72    	[ -1.41977881e-02   1.19980000e+02]	[  2.70045523e-04   6.64677365e+00]	[ -1.51414120e-02   1.06000000e+02]	[ -1.37148743e-02   1.44000000e+02]
    11 	62    	[ -1.40695285e-02   1.23770000e+02]	[  2.85048827e-04   7.69656417e+00]	[ -1.53651584e-02   1.05000000e+02]	[ -1.36951864e-02   1.44000000e+02]
    12 	70    	[ -1.39961053e-02   1.26310000e+02]	[  4.43489853e-04   7.26869314e+00]	[ -1.77504849e-02   1.06000000e+02]	[ -1.36474687e-02   1.44000000e+02]
    13 	75    	[ -1.38600038e-02   1.28330000e+02]	[  2.52901473e-04   6.85281694e+00]	[ -1.55939005e-02   1.05000000e+02]	[ -1.36107906e-02   1.45000000e+02]
    14 	81    	[ -1.37694032e-02   1.28780000e+02]	[  1.57513515e-04   5.71065670e+00]	[ -1.46090456e-02   1.14000000e+02]	[ -1.35778299e-02   1.44000000e+02]
    15 	83    	[ -1.37912772e-02   1.28670000e+02]	[  3.71558400e-04   4.57614467e+00]	[ -1.65989184e-02   1.18000000e+02]	[ -1.35409180e-02   1.39000000e+02]
    16 	72    	[ -1.37216783e-02   1.29050000e+02]	[  2.33279829e-04   3.93287427e+00]	[ -1.49458220e-02   1.22000000e+02]	[ -1.35260133e-02   1.39000000e+02]
    17 	74    	[ -1.37015927e-02   1.28890000e+02]	[  1.98963136e-04   4.41337739e+00]	[ -1.45991258e-02   1.20000000e+02]	[ -1.34919965e-02   1.41000000e+02]
    18 	78    	[ -1.36553269e-02   1.28380000e+02]	[  2.49047189e-04   4.48727089e+00]	[ -1.55110901e-02   1.17000000e+02]	[ -1.34620789e-02   1.41000000e+02]
    19 	74    	[ -1.36875677e-02   1.27750000e+02]	[  3.40448288e-04   4.12401503e+00]	[ -1.51555179e-02   1.17000000e+02]	[ -1.34204248e-02   1.37000000e+02]
    20 	60    	[ -1.35688749e-02   1.27560000e+02]	[  1.92650903e-04   4.29725494e+00]	[ -1.47728838e-02   1.17000000e+02]	[ -1.33748087e-02   1.37000000e+02]
    21 	83    	[ -1.35613940e-02   1.27630000e+02]	[  2.46874291e-04   4.59707516e+00]	[ -1.48400730e-02   1.15000000e+02]	[ -1.33748087e-02   1.40000000e+02]
    22 	82    	[ -1.35623181e-02   1.26950000e+02]	[  2.99719146e-04   4.16263138e+00]	[ -1.49933071e-02   1.15000000e+02]	[ -1.33605999e-02   1.34000000e+02]
    23 	78    	[ -1.35458327e-02   1.26530000e+02]	[  3.41513654e-04   4.46867989e+00]	[ -1.57294540e-02   1.17000000e+02]	[ -1.33550302e-02   1.36000000e+02]
    24 	88    	[ -1.35881455e-02   1.25250000e+02]	[  7.20465660e-04   4.25764019e+00]	[ -2.00587334e-02   1.11000000e+02]	[ -1.33299520e-02   1.34000000e+02]
    25 	80    	[ -1.35860755e-02   1.24420000e+02]	[  4.10951164e-04   4.67585286e+00]	[ -1.59004873e-02   1.11000000e+02]	[ -1.33299520e-02   1.35000000e+02]
    26 	80    	[ -1.36192980e-02   1.23360000e+02]	[  7.34757442e-04   4.08049017e+00]	[ -1.88924422e-02   1.10000000e+02]	[ -1.33147791e-02   1.30000000e+02]
    27 	60    	[ -1.34423872e-02   1.22630000e+02]	[  2.30360678e-04   3.81485255e+00]	[ -1.46303585e-02   1.13000000e+02]	[ -1.32843116e-02   1.33000000e+02]
    28 	78    	[ -1.34970301e-02   1.21180000e+02]	[  3.97822493e-04   3.93288698e+00]	[ -1.54323284e-02   1.09000000e+02]	[ -1.32843116e-02   1.30000000e+02]
    29 	66    	[ -1.34489061e-02   1.20030000e+02]	[  4.11053801e-04   4.04834534e+00]	[ -1.58811294e-02   1.08000000e+02]	[ -1.32608577e-02   1.29000000e+02]
    30 	76    	[ -1.33852887e-02   1.19360000e+02]	[  1.72134955e-04   3.67564960e+00]	[ -1.43205213e-02   1.08000000e+02]	[ -1.32519177e-02   1.29000000e+02]
    31 	79    	[ -1.34164693e-02   1.19220000e+02]	[  2.88500504e-04   3.60438622e+00]	[ -1.52829809e-02   1.08000000e+02]	[ -1.32492470e-02   1.31000000e+02]
    32 	81    	[ -1.34001161e-02   1.18720000e+02]	[  2.51032977e-04   3.41490849e+00]	[ -1.48833480e-02   1.11000000e+02]	[ -1.32453952e-02   1.27000000e+02]
    33 	75    	[ -1.33805788e-02   1.17640000e+02]	[  3.00205313e-04   3.11294073e+00]	[ -1.55866805e-02   1.11000000e+02]	[ -1.32453952e-02   1.26000000e+02]
    34 	81    	[ -1.34261646e-02   1.16620000e+02]	[  4.81908496e-04   3.53491160e+00]	[ -1.60529809e-02   1.08000000e+02]	[ -1.32265167e-02   1.27000000e+02]
    35 	76    	[ -1.34477969e-02   1.16880000e+02]	[  7.05282493e-04   3.37425547e+00]	[ -1.91819318e-02   1.07000000e+02]	[ -1.31991079e-02   1.25000000e+02]
    36 	78    	[ -1.33990178e-02   1.16740000e+02]	[  5.53045996e-04   3.50034284e+00]	[ -1.82113919e-02   1.07000000e+02]	[ -1.31914118e-02   1.26000000e+02]
    37 	84    	[ -1.33573502e-02   1.15830000e+02]	[  3.13917882e-04   3.77108738e+00]	[ -1.50366997e-02   1.05000000e+02]	[ -1.31663287e-02   1.23000000e+02]
    38 	77    	[ -1.33781638e-02   1.16050000e+02]	[  3.44485137e-04   4.47297440e+00]	[ -1.52407895e-02   9.90000000e+01]	[ -1.31821369e-02   1.24000000e+02]
    39 	64    	[ -1.33044905e-02   1.15590000e+02]	[  2.40348372e-04   4.16676133e+00]	[ -1.45812121e-02   1.03000000e+02]	[ -1.31707440e-02   1.26000000e+02]
    40 	74    	[ -1.32909746e-02   1.13970000e+02]	[  2.64637816e-04   3.93815947e+00]	[ -1.47948515e-02   1.03000000e+02]	[ -1.31678643e-02   1.26000000e+02]
    41 	76    	[ -1.33336360e-02   1.13220000e+02]	[  4.62754899e-04   3.34837274e+00]	[ -1.62300702e-02   1.05000000e+02]	[ -1.31573367e-02   1.24000000e+02]
    42 	73    	[ -1.32844407e-02   1.12590000e+02]	[  3.01796122e-04   3.61135709e+00]	[ -1.55364257e-02   1.05000000e+02]	[ -1.31441402e-02   1.24000000e+02]
    43 	70    	[ -1.33148326e-02   1.11910000e+02]	[  3.07499046e-04   3.03016501e+00]	[ -1.46481738e-02   1.03000000e+02]	[ -1.31339338e-02   1.21000000e+02]
    44 	79    	[ -1.32607492e-02   1.11680000e+02]	[  2.61963966e-04   3.43185081e+00]	[ -1.49158014e-02   1.03000000e+02]	[ -1.31286725e-02   1.21000000e+02]
    45 	60    	[ -1.32464113e-02   1.11050000e+02]	[  2.33657633e-04   3.37453701e+00]	[ -1.42633295e-02   1.00000000e+02]	[ -1.31286725e-02   1.19000000e+02]
    46 	77    	[ -1.32625344e-02   1.10170000e+02]	[  3.04703701e-04   2.94976270e+00]	[ -1.46988086e-02   1.01000000e+02]	[ -1.31236334e-02   1.20000000e+02]
    47 	68    	[ -1.32370843e-02   1.10260000e+02]	[  3.19593975e-04   2.75905781e+00]	[ -1.50342317e-02   1.02000000e+02]	[ -1.31143622e-02   1.19000000e+02]
    48 	83    	[ -1.32677096e-02   1.08990000e+02]	[  3.86156712e-04   2.38535113e+00]	[ -1.56974968e-02   1.01000000e+02]	[ -1.31151680e-02   1.15000000e+02]
    49 	79    	[ -1.32525353e-02   1.08480000e+02]	[  4.58734955e-04   2.22027025e+00]	[ -1.71864389e-02   1.01000000e+02]	[ -1.31150939e-02   1.13000000e+02]
    50 	63    	[ -1.32037460e-02   1.08430000e+02]	[  2.99277829e-04   1.78468485e+00]	[ -1.56687263e-02   1.04000000e+02]	[ -1.31135419e-02   1.13000000e+02]
    51 	72    	[ -1.32304137e-02   1.07610000e+02]	[  2.98582840e-04   2.11137396e+00]	[ -1.50113380e-02   1.01000000e+02]	[ -1.31137671e-02   1.13000000e+02]
    52 	71    	[ -1.32574072e-02   1.07640000e+02]	[  4.42552838e-04   2.37705700e+00]	[ -1.59953388e-02   1.00000000e+02]	[ -1.31137671e-02   1.16000000e+02]
    53 	75    	[ -1.32416354e-02   1.07470000e+02]	[  3.32533045e-04   2.24702025e+00]	[ -1.53188635e-02   1.01000000e+02]	[ -1.31137671e-02   1.13000000e+02]
    54 	86    	[ -1.31816517e-02   1.07710000e+02]	[  1.92530677e-04   2.21492664e+00]	[ -1.43543488e-02   1.02000000e+02]	[ -1.31108106e-02   1.16000000e+02]
    55 	79    	[ -1.32705337e-02   1.08140000e+02]	[  6.69136540e-04   1.86558302e+00]	[ -1.93734077e-02   1.02000000e+02]	[ -1.31014344e-02   1.12000000e+02]
    56 	80    	[ -1.32210279e-02   1.08030000e+02]	[  3.21955639e-04   1.95680863e+00]	[ -1.54049384e-02   1.02000000e+02]	[ -1.31014344e-02   1.16000000e+02]
    57 	80    	[ -1.32215638e-02   1.07400000e+02]	[  3.12699264e-04   2.24499443e+00]	[ -1.46297252e-02   1.00000000e+02]	[ -1.30998825e-02   1.14000000e+02]
    58 	78    	[ -1.32830004e-02   1.07150000e+02]	[  4.67583048e-04   2.11837202e+00]	[ -1.58468824e-02   1.01000000e+02]	[ -1.30992528e-02   1.14000000e+02]
    59 	72    	[ -1.32057259e-02   1.07020000e+02]	[  2.81985556e-04   2.29773802e+00]	[ -1.44083216e-02   1.01000000e+02]	[ -1.30990373e-02   1.16000000e+02]
    60 	79    	[ -1.31781282e-02   1.06900000e+02]	[  1.72653034e-04   1.90525589e+00]	[ -1.40709826e-02   1.03000000e+02]	[ -1.30867355e-02   1.12000000e+02]
    61 	75    	[ -1.32659234e-02   1.06410000e+02]	[  6.22431047e-04   2.07891799e+00]	[ -1.84745471e-02   1.02000000e+02]	[ -1.30988013e-02   1.11000000e+02]
    62 	78    	[ -1.32022115e-02   1.06010000e+02]	[  3.10436912e-04   2.49998000e+00]	[ -1.51523389e-02   1.00000000e+02]	[ -1.30989502e-02   1.16000000e+02]
    63 	72    	[ -1.31917881e-02   1.06430000e+02]	[  2.50824593e-04   2.05550480e+00]	[ -1.45287092e-02   1.01000000e+02]	[ -1.30989502e-02   1.15000000e+02]
    64 	76    	[ -1.31681349e-02   1.06310000e+02]	[  2.60899135e-04   1.77028246e+00]	[ -1.49123341e-02   1.01000000e+02]	[ -1.30955561e-02   1.13000000e+02]
    65 	78    	[ -1.32134338e-02   1.06430000e+02]	[  2.79460962e-04   1.89871009e+00]	[ -1.43847849e-02   9.70000000e+01]	[ -1.30951990e-02   1.12000000e+02]
    66 	84    	[ -1.32074215e-02   1.05900000e+02]	[  2.67088612e-04   1.64620776e+00]	[ -1.41645131e-02   1.01000000e+02]	[ -1.30951990e-02   1.09000000e+02]
    67 	74    	[ -1.31825633e-02   1.06340000e+02]	[  3.25206227e-04   2.20099977e+00]	[ -1.59827904e-02   1.01000000e+02]	[ -1.30937625e-02   1.15000000e+02]
    68 	78    	[ -1.32252991e-02   1.05960000e+02]	[  3.67529903e-04   1.84889156e+00]	[ -1.60382255e-02   1.01000000e+02]	[ -1.30937625e-02   1.12000000e+02]
    69 	79    	[ -1.32516054e-02   1.06440000e+02]	[  3.85690952e-04   2.10864886e+00]	[ -1.49337151e-02   1.00000000e+02]	[ -1.30884075e-02   1.14000000e+02]
    70 	71    	[ -1.32233623e-02   1.06030000e+02]	[  3.12841358e-04   2.42674679e+00]	[ -1.50879145e-02   9.70000000e+01]	[ -1.30884075e-02   1.12000000e+02]
    71 	79    	[ -1.33073826e-02   1.06010000e+02]	[  7.78597350e-04   2.25608067e+00]	[ -2.00725685e-02   1.02000000e+02]	[ -1.30884074e-02   1.16000000e+02]
    72 	75    	[ -1.31913364e-02   1.06250000e+02]	[  2.94977944e-04   2.46322959e+00]	[ -1.48102344e-02   9.60000000e+01]	[ -1.30884074e-02   1.13000000e+02]
    73 	76    	[ -1.32962776e-02   1.07320000e+02]	[  4.43824368e-04   2.40366387e+00]	[ -1.57271565e-02   9.80000000e+01]	[ -1.30778174e-02   1.12000000e+02]
    74 	78    	[ -1.32376377e-02   1.08040000e+02]	[  6.01811452e-04   2.34912750e+00]	[ -1.85912800e-02   1.03000000e+02]	[ -1.30672534e-02   1.18000000e+02]
    75 	78    	[ -1.31855234e-02   1.08180000e+02]	[  3.07020916e-04   2.03164958e+00]	[ -1.49239475e-02   1.00000000e+02]	[ -1.30672534e-02   1.14000000e+02]
    76 	72    	[ -1.31846144e-02   1.07570000e+02]	[  4.06916996e-04   1.75644527e+00]	[ -1.64227611e-02   1.03000000e+02]	[ -1.30672534e-02   1.12000000e+02]
    77 	71    	[ -1.32580774e-02   1.07560000e+02]	[  3.94936339e-04   2.63939387e+00]	[ -1.52017686e-02   9.80000000e+01]	[ -1.30672534e-02   1.14000000e+02]
    78 	72    	[ -1.32361776e-02   1.07380000e+02]	[  3.61985241e-04   2.42396370e+00]	[ -1.46817624e-02   1.02000000e+02]	[ -1.30619076e-02   1.15000000e+02]
    79 	82    	[ -1.31921127e-02   1.07180000e+02]	[  4.54315300e-04   1.74000000e+00]	[ -1.68754791e-02   1.04000000e+02]	[ -1.30519349e-02   1.13000000e+02]
    80 	87    	[ -1.31585857e-02   1.06540000e+02]	[  2.20243841e-04   2.00708744e+00]	[ -1.40244464e-02   9.90000000e+01]	[ -1.30509347e-02   1.13000000e+02]
    81 	85    	[ -1.32318727e-02   1.06830000e+02]	[  3.96262752e-04   2.24078111e+00]	[ -1.49001011e-02   1.01000000e+02]	[ -1.30509347e-02   1.14000000e+02]
    82 	61    	[ -1.31743542e-02   1.06680000e+02]	[  2.73919974e-04   2.12546466e+00]	[ -1.46683216e-02   1.03000000e+02]	[ -1.30509347e-02   1.14000000e+02]
    83 	67    	[ -1.31292598e-02   1.06580000e+02]	[  2.10273308e-04   2.37983193e+00]	[ -1.42440224e-02   9.80000000e+01]	[ -1.30470031e-02   1.13000000e+02]
    84 	85    	[ -1.31803633e-02   1.05990000e+02]	[  3.13693232e-04   2.37274103e+00]	[ -1.49460909e-02   1.00000000e+02]	[ -1.30450980e-02   1.16000000e+02]
    85 	82    	[ -1.32529920e-02   1.05150000e+02]	[  5.48136981e-04   2.22429764e+00]	[ -1.65184714e-02   1.00000000e+02]	[ -1.30421949e-02   1.13000000e+02]
    86 	79    	[ -1.31644807e-02   1.05080000e+02]	[  3.08085558e-04   2.04293906e+00]	[ -1.50879431e-02   9.90000000e+01]	[ -1.30421949e-02   1.12000000e+02]
    87 	77    	[ -1.31772258e-02   1.04720000e+02]	[  4.19851060e-04   2.20490363e+00]	[ -1.52441206e-02   9.80000000e+01]	[ -1.30421949e-02   1.11000000e+02]
    88 	82    	[ -1.31878238e-02   1.04530000e+02]	[  3.59301923e-04   2.10454271e+00]	[ -1.52907260e-02   9.90000000e+01]	[ -1.30421949e-02   1.12000000e+02]
    89 	77    	[ -1.31106736e-02   1.04060000e+02]	[  1.93227097e-04   2.28394396e+00]	[ -1.40480636e-02   9.70000000e+01]	[ -1.30421949e-02   1.11000000e+02]
    90 	82    	[ -1.31523540e-02   1.03380000e+02]	[  2.91190644e-04   2.21711524e+00]	[ -1.47297190e-02   9.60000000e+01]	[ -1.30421949e-02   1.09000000e+02]
    91 	75    	[ -1.31719592e-02   1.03130000e+02]	[  3.22954174e-04   2.30067381e+00]	[ -1.45497648e-02   9.80000000e+01]	[ -1.30421949e-02   1.12000000e+02]
    92 	76    	[ -1.31551837e-02   1.02750000e+02]	[  3.39953517e-04   2.14184500e+00]	[ -1.52875378e-02   9.60000000e+01]	[ -1.30421949e-02   1.13000000e+02]
    93 	81    	[ -1.31636499e-02   1.03010000e+02]	[  3.60987868e-04   2.32161582e+00]	[ -1.55671542e-02   9.80000000e+01]	[ -1.30421949e-02   1.12000000e+02]
    94 	80    	[ -1.31266697e-02   1.03180000e+02]	[  2.46532742e-04   2.05611284e+00]	[ -1.43382461e-02   9.80000000e+01]	[ -1.30421949e-02   1.11000000e+02]
    95 	69    	[ -1.31179566e-02   1.03300000e+02]	[  2.26410727e-04   1.90525589e+00]	[ -1.43409908e-02   9.70000000e+01]	[ -1.30421949e-02   1.09000000e+02]
    96 	73    	[ -1.31436161e-02   1.03830000e+02]	[  2.75679877e-04   2.16820202e+00]	[ -1.49177741e-02   9.60000000e+01]	[ -1.30421949e-02   1.10000000e+02]
    97 	83    	[ -1.31666229e-02   1.04010000e+02]	[  3.40466884e-04   1.93129490e+00]	[ -1.48352758e-02   9.80000000e+01]	[ -1.30385514e-02   1.11000000e+02]
    98 	84    	[ -1.32401922e-02   1.03920000e+02]	[  4.81817854e-04   1.85299757e+00]	[ -1.60508676e-02   9.80000000e+01]	[ -1.30385514e-02   1.10000000e+02]
    99 	74    	[ -1.31000318e-02   1.04070000e+02]	[  1.73370812e-04   1.36568664e+00]	[ -1.43277828e-02   1.00000000e+02]	[ -1.30385514e-02   1.08000000e+02]
    100	86    	[ -1.31473731e-02   1.03820000e+02]	[  4.47234243e-04   1.50585524e+00]	[ -1.70451883e-02   9.80000000e+01]	[ -1.30385514e-02   1.08000000e+02]
    101	70    	[ -1.31162380e-02   1.03390000e+02]	[  2.39216464e-04   1.74295726e+00]	[ -1.48821607e-02   9.70000000e+01]	[ -1.30344146e-02   1.11000000e+02]
    102	74    	[ -1.31499559e-02   1.02830000e+02]	[  3.45931591e-04   1.54954832e+00]	[ -1.51737181e-02   9.90000000e+01]	[ -1.30344146e-02   1.10000000e+02]
    103	74    	[ -1.31787945e-02   1.02390000e+02]	[  6.15810707e-04   1.55495981e+00]	[ -1.86932693e-02   9.80000000e+01]	[ -1.30344146e-02   1.08000000e+02]
    104	75    	[ -1.31275680e-02   1.02260000e+02]	[  2.62832876e-04   1.70070574e+00]	[ -1.51271966e-02   9.60000000e+01]	[ -1.30342867e-02   1.08000000e+02]
    105	84    	[ -1.31712149e-02   1.02370000e+02]	[  3.98837846e-04   1.94244691e+00]	[ -1.56103511e-02   9.80000000e+01]	[ -1.30341789e-02   1.10000000e+02]
    106	77    	[ -1.31080893e-02   1.01860000e+02]	[  2.33401080e-04   1.53635933e+00]	[ -1.45409128e-02   9.80000000e+01]	[ -1.30341789e-02   1.07000000e+02]
    107	80    	[ -1.31244563e-02   1.01130000e+02]	[  3.11861060e-04   1.73582833e+00]	[ -1.48451473e-02   9.50000000e+01]	[ -1.30341789e-02   1.07000000e+02]
    108	82    	[ -1.31129540e-02   1.00660000e+02]	[  2.42167787e-04   1.63841387e+00]	[ -1.45078082e-02   9.70000000e+01]	[ -1.30341789e-02   1.08000000e+02]
    109	68    	[ -1.31813446e-02   1.00300000e+02]	[  5.92903924e-04   1.59059737e+00]	[ -1.80113163e-02   9.70000000e+01]	[ -1.30341789e-02   1.10000000e+02]
    110	78    	[ -1.31421182e-02   9.96600000e+01]	[  2.96499987e-04   1.43680200e+00]	[ -1.46147015e-02   9.30000000e+01]	[ -1.30341789e-02   1.06000000e+02]
    111	77    	[ -1.31890032e-02   9.91200000e+01]	[  4.25292978e-04   1.62653005e+00]	[ -1.54536545e-02   9.40000000e+01]	[ -1.30341789e-02   1.07000000e+02]
    112	79    	[ -1.32261442e-02   9.89100000e+01]	[  4.73704119e-04   2.01044771e+00]	[ -1.56504966e-02   9.30000000e+01]	[ -1.30341789e-02   1.05000000e+02]
    113	81    	[ -1.31795682e-02   9.82800000e+01]	[  3.62946769e-04   1.79488161e+00]	[ -1.47805203e-02   9.00000000e+01]	[ -1.30341789e-02   1.06000000e+02]
    114	77    	[ -1.31887809e-02   9.81700000e+01]	[  3.18167147e-04   2.02017326e+00]	[ -1.48646993e-02   9.20000000e+01]	[ -1.30341789e-02   1.06000000e+02]
    115	77    	[ -1.32452101e-02   9.82900000e+01]	[  7.19304393e-04   1.79050272e+00]	[ -1.88994808e-02   9.10000000e+01]	[ -1.30341789e-02   1.04000000e+02]
    116	81    	[ -1.30991491e-02   9.82000000e+01]	[  2.10404927e-04   1.33416641e+00]	[ -1.46740390e-02   9.30000000e+01]	[ -1.30341789e-02   1.03000000e+02]
    117	78    	[ -1.31338328e-02   9.82300000e+01]	[  3.89019446e-04   9.98548947e-01]	[ -1.52334616e-02   9.70000000e+01]	[ -1.30341789e-02   1.04000000e+02]
    118	73    	[ -1.31534604e-02   9.84500000e+01]	[  3.17437266e-04   1.35185058e+00]	[ -1.44462284e-02   9.40000000e+01]	[ -1.30341789e-02   1.04000000e+02]
    119	67    	[ -1.32067020e-02   9.82700000e+01]	[  6.96777074e-04   1.95374000e+00]	[ -1.95467420e-02   9.10000000e+01]	[ -1.30341789e-02   1.09000000e+02]
    120	71    	[ -1.31460345e-02   9.83500000e+01]	[  3.16963721e-04   1.59608897e+00]	[ -1.49207185e-02   9.30000000e+01]	[ -1.30341789e-02   1.06000000e+02]
    121	81    	[ -1.31016144e-02   9.82900000e+01]	[  2.06189320e-04   1.35863902e+00]	[ -1.41559320e-02   9.40000000e+01]	[ -1.30341789e-02   1.06000000e+02]
    122	82    	[ -1.31220074e-02   9.83600000e+01]	[  3.30623180e-04   1.83040979e+00]	[ -1.52553993e-02   9.10000000e+01]	[ -1.30341789e-02   1.08000000e+02]
    123	74    	[ -1.31978705e-02   9.83900000e+01]	[  3.71122169e-04   2.07795573e+00]	[ -1.52267638e-02   9.20000000e+01]	[ -1.30341789e-02   1.10000000e+02]
    124	68    	[ -1.31059760e-02   9.80100000e+01]	[  1.99498928e-04   1.14450863e+00]	[ -1.40300737e-02   9.30000000e+01]	[ -1.30341789e-02   1.04000000e+02]
    125	77    	[ -1.31260812e-02   9.80300000e+01]	[  2.92671589e-04   1.27636202e+00]	[ -1.46589792e-02   9.30000000e+01]	[ -1.30341789e-02   1.06000000e+02]
    126	77    	[ -1.31720931e-02   9.84300000e+01]	[  3.55659270e-04   1.56368155e+00]	[ -1.49518513e-02   9.40000000e+01]	[ -1.30341789e-02   1.05000000e+02]
    127	77    	[ -1.30954599e-02   9.84400000e+01]	[  1.46366065e-04   1.76249823e+00]	[ -1.37553757e-02   9.60000000e+01]	[ -1.30341789e-02   1.08000000e+02]
    128	75    	[ -1.31372847e-02   9.82000000e+01]	[  3.00331199e-04   1.05830052e+00]	[ -1.45955138e-02   9.30000000e+01]	[ -1.30341789e-02   1.04000000e+02]
    129	84    	[ -1.31831442e-02   9.81800000e+01]	[  4.38553583e-04   1.40982268e+00]	[ -1.64259938e-02   9.20000000e+01]	[ -1.30341789e-02   1.04000000e+02]
    130	76    	[ -1.31436006e-02   9.83400000e+01]	[  3.26994281e-04   1.30552671e+00]	[ -1.55291268e-02   9.60000000e+01]	[ -1.30341789e-02   1.04000000e+02]
    131	68    	[ -1.31673081e-02   9.84200000e+01]	[  3.71638884e-04   1.45725770e+00]	[ -1.56037372e-02   9.50000000e+01]	[ -1.30341789e-02   1.05000000e+02]
    132	77    	[ -1.31250352e-02   9.85400000e+01]	[  2.29899579e-04   1.81339461e+00]	[ -1.43767982e-02   9.40000000e+01]	[ -1.30341789e-02   1.06000000e+02]
    133	80    	[ -1.31032074e-02   9.82500000e+01]	[  1.97452251e-04   1.37386317e+00]	[ -1.42683750e-02   9.20000000e+01]	[ -1.30341789e-02   1.07000000e+02]
    134	76    	[ -1.32233516e-02   9.82500000e+01]	[  4.19405795e-04   1.84594149e+00]	[ -1.52602136e-02   9.30000000e+01]	[ -1.30341789e-02   1.05000000e+02]
    135	83    	[ -1.31376073e-02   9.83300000e+01]	[  2.47345270e-04   1.54954832e+00]	[ -1.42418353e-02   9.40000000e+01]	[ -1.30341789e-02   1.06000000e+02]
    136	73    	[ -1.30806073e-02   9.82500000e+01]	[  1.59055979e-04   1.16940156e+00]	[ -1.40261796e-02   9.50000000e+01]	[ -1.30341789e-02   1.05000000e+02]
    137	72    	[ -1.31340718e-02   9.81400000e+01]	[  3.26047639e-04   1.43540935e+00]	[ -1.51148076e-02   9.20000000e+01]	[ -1.30341789e-02   1.05000000e+02]
    138	72    	[ -1.31257742e-02   9.80400000e+01]	[  2.94551376e-04   1.07628992e+00]	[ -1.51720752e-02   9.40000000e+01]	[ -1.30341789e-02   1.03000000e+02]
    139	65    	[ -1.31501117e-02   9.86200000e+01]	[  2.70326107e-04   2.12499412e+00]	[ -1.40765910e-02   9.50000000e+01]	[ -1.30341789e-02   1.10000000e+02]
    140	85    	[ -1.31983478e-02   9.79400000e+01]	[  4.49991671e-04   1.76533283e+00]	[ -1.52889410e-02   9.20000000e+01]	[ -1.30341789e-02   1.05000000e+02]
    141	72    	[ -1.31204031e-02   9.81500000e+01]	[  3.03433293e-04   1.55804365e+00]	[ -1.52085113e-02   9.20000000e+01]	[ -1.30293771e-02   1.06000000e+02]
    142	83    	[ -1.31328373e-02   9.81100000e+01]	[  3.30204560e-04   9.15368778e-01]	[ -1.51344213e-02   9.50000000e+01]	[ -1.30293771e-02   1.02000000e+02]
    143	82    	[ -1.31394935e-02   9.79900000e+01]	[  3.32772738e-04   1.38199132e+00]	[ -1.57235150e-02   9.40000000e+01]	[ -1.30293771e-02   1.04000000e+02]
    144	82    	[ -1.31550851e-02   9.78100000e+01]	[  3.47167404e-04   1.50794562e+00]	[ -1.49896196e-02   9.40000000e+01]	[ -1.30293771e-02   1.03000000e+02]
    145	79    	[ -1.30891681e-02   9.74700000e+01]	[  1.80327301e-04   1.60284123e+00]	[ -1.44426995e-02   9.10000000e+01]	[ -1.30293771e-02   1.04000000e+02]
    146	79    	[ -1.31539376e-02   9.71800000e+01]	[  3.14767028e-04   1.24402572e+00]	[ -1.45627170e-02   9.10000000e+01]	[ -1.30293771e-02   1.01000000e+02]
    147	80    	[ -1.31221684e-02   9.72900000e+01]	[  2.65271368e-04   1.43034961e+00]	[ -1.48698052e-02   9.40000000e+01]	[ -1.30293771e-02   1.04000000e+02]
    148	77    	[ -1.31224649e-02   9.74500000e+01]	[  2.91257102e-04   1.50582203e+00]	[ -1.49468664e-02   9.20000000e+01]	[ -1.30293771e-02   1.03000000e+02]
    149	71    	[ -1.30995582e-02   9.73000000e+01]	[  2.09448485e-04   1.72336879e+00]	[ -1.40669103e-02   9.30000000e+01]	[ -1.30293771e-02   1.07000000e+02]
    150	83    	[ -1.31549930e-02   9.75600000e+01]	[  3.28214339e-04   1.85105375e+00]	[ -1.48699858e-02   9.30000000e+01]	[ -1.30219994e-02   1.04000000e+02]
    151	82    	[ -1.31652513e-02   9.74000000e+01]	[  4.09482885e-04   1.32664992e+00]	[ -1.56393126e-02   9.50000000e+01]	[ -1.30219994e-02   1.03000000e+02]
    152	74    	[ -1.32338500e-02   9.70200000e+01]	[  6.96453025e-04   1.31133520e+00]	[ -1.89266745e-02   9.00000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    153	72    	[ -1.31473705e-02   9.73300000e+01]	[  3.76492975e-04   1.56878934e+00]	[ -1.58854512e-02   9.50000000e+01]	[ -1.30216692e-02   1.05000000e+02]
    154	74    	[ -1.31226862e-02   9.69600000e+01]	[  2.66704491e-04   1.63046006e+00]	[ -1.47587189e-02   9.40000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    155	80    	[ -1.31502722e-02   9.68300000e+01]	[  3.07417305e-04   2.05453158e+00]	[ -1.44374604e-02   9.10000000e+01]	[ -1.30216692e-02   1.06000000e+02]
    156	76    	[ -1.31577141e-02   9.62200000e+01]	[  4.41490740e-04   1.64060964e+00]	[ -1.63477836e-02   8.70000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    157	66    	[ -1.31182705e-02   9.62700000e+01]	[  3.86501121e-04   1.22356038e+00]	[ -1.63947290e-02   9.20000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    158	73    	[ -1.32784062e-02   9.63600000e+01]	[  9.46023746e-04   1.32302683e+00]	[ -1.95296401e-02   9.40000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    159	76    	[ -1.31251540e-02   9.64600000e+01]	[  2.71177291e-04   1.62739055e+00]	[ -1.44335429e-02   9.20000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    160	74    	[ -1.30889210e-02   9.63300000e+01]	[  1.78179958e-04   1.36422139e+00]	[ -1.40204798e-02   9.20000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    161	70    	[ -1.32094826e-02   9.63400000e+01]	[  5.91708748e-04   1.68653491e+00]	[ -1.82611605e-02   9.20000000e+01]	[ -1.30216692e-02   1.05000000e+02]
    162	68    	[ -1.31451348e-02   9.64700000e+01]	[  3.19372375e-04   1.65804101e+00]	[ -1.46159126e-02   9.30000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    163	75    	[ -1.31583700e-02   9.63600000e+01]	[  6.22356828e-04   1.60947196e+00]	[ -1.89383215e-02   9.00000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    164	81    	[ -1.31770887e-02   9.62200000e+01]	[  4.40994538e-04   1.44623650e+00]	[ -1.60799861e-02   9.00000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    165	70    	[ -1.31439959e-02   9.63200000e+01]	[  3.57296473e-04   2.02918703e+00]	[ -1.48710861e-02   8.90000000e+01]	[ -1.30216692e-02   1.06000000e+02]
    166	83    	[ -1.32317062e-02   9.63500000e+01]	[  7.41622763e-04   2.25554871e+00]	[ -1.96277465e-02   9.00000000e+01]	[ -1.30216692e-02   1.05000000e+02]
    167	78    	[ -1.31080652e-02   9.61300000e+01]	[  2.85333850e-04   1.76439791e+00]	[ -1.49429365e-02   9.00000000e+01]	[ -1.30216692e-02   1.06000000e+02]
    168	79    	[ -1.31253062e-02   9.63100000e+01]	[  3.13984570e-04   1.94265283e+00]	[ -1.48963184e-02   9.10000000e+01]	[ -1.30216692e-02   1.08000000e+02]
    169	87    	[ -1.31130769e-02   9.64600000e+01]	[  2.86022400e-04   1.59637088e+00]	[ -1.45894355e-02   9.10000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    170	77    	[ -1.31274764e-02   9.64800000e+01]	[  3.08316802e-04   1.78033705e+00]	[ -1.52101848e-02   9.20000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    171	83    	[ -1.31488150e-02   9.62000000e+01]	[  3.28191419e-04   1.80554701e+00]	[ -1.46930830e-02   9.10000000e+01]	[ -1.30214104e-02   1.04000000e+02]
    172	83    	[ -1.31331070e-02   9.57400000e+01]	[  3.23458310e-04   1.87946801e+00]	[ -1.49232478e-02   8.80000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    173	75    	[ -1.32129278e-02   9.57600000e+01]	[  7.81765024e-04   1.97038067e+00]	[ -2.04242265e-02   8.50000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    174	80    	[ -1.31805822e-02   9.52500000e+01]	[  4.42433954e-04   2.01680440e+00]	[ -1.57411702e-02   8.80000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    175	77    	[ -1.31422309e-02   9.54300000e+01]	[  3.75001889e-04   1.57006369e+00]	[ -1.50213869e-02   9.00000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    176	77    	[ -1.31687829e-02   9.52200000e+01]	[  3.67030174e-04   1.55293271e+00]	[ -1.49543081e-02   9.10000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    177	71    	[ -1.31291667e-02   9.53100000e+01]	[  2.54837777e-04   1.45392572e+00]	[ -1.45424809e-02   9.20000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    178	72    	[ -1.31110758e-02   9.52100000e+01]	[  2.50486805e-04   1.49863271e+00]	[ -1.42416079e-02   8.80000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    179	65    	[ -1.31111290e-02   9.53600000e+01]	[  2.73188495e-04   1.50013333e+00]	[ -1.46064588e-02   9.20000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    180	72    	[ -1.31626479e-02   9.53400000e+01]	[  3.92751194e-04   1.76193076e+00]	[ -1.52424355e-02   8.90000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    181	72    	[ -1.31235821e-02   9.52800000e+01]	[  3.09720864e-04   1.59423963e+00]	[ -1.48765833e-02   8.80000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    182	75    	[ -1.31719030e-02   9.51900000e+01]	[  3.06297519e-04   1.54722332e+00]	[ -1.43647219e-02   9.00000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    183	78    	[ -1.31930189e-02   9.51500000e+01]	[  4.02982609e-04   1.52561463e+00]	[ -1.47955860e-02   9.10000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    184	77    	[ -1.32478820e-02   9.53600000e+01]	[  7.35091406e-04   1.64632925e+00]	[ -1.95049833e-02   9.10000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    185	71    	[ -1.31338409e-02   9.55400000e+01]	[  3.09561721e-04   1.53896069e+00]	[ -1.47482087e-02   9.20000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    186	81    	[ -1.31248123e-02   9.56100000e+01]	[  3.20071827e-04   2.04887774e+00]	[ -1.49640040e-02   9.20000000e+01]	[ -1.30216692e-02   1.06000000e+02]
    187	73    	[ -1.31269914e-02   9.52500000e+01]	[  3.12415786e-04   1.45172311e+00]	[ -1.47430028e-02   9.00000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    188	77    	[ -1.31983940e-02   9.54100000e+01]	[  3.78844447e-04   2.01044771e+00]	[ -1.45053346e-02   8.70000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    189	79    	[ -1.31565490e-02   9.54900000e+01]	[  3.89812908e-04   1.88941790e+00]	[ -1.55759727e-02   9.00000000e+01]	[ -1.30216692e-02   1.06000000e+02]
    190	77    	[ -1.31694392e-02   9.57600000e+01]	[  3.64075671e-04   2.03036942e+00]	[ -1.50592047e-02   8.80000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    191	69    	[ -1.32689892e-02   9.54400000e+01]	[  9.25866946e-04   1.73389734e+00]	[ -1.99998573e-02   9.10000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    192	85    	[ -1.31727572e-02   9.53600000e+01]	[  3.83574940e-04   1.55897402e+00]	[ -1.50358566e-02   8.80000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    193	72    	[ -1.30932614e-02   9.54600000e+01]	[  2.19556375e-04   1.35955875e+00]	[ -1.47497432e-02   9.30000000e+01]	[ -1.30216692e-02   1.00000000e+02]
    194	81    	[ -1.31515393e-02   9.52500000e+01]	[  3.34711696e-04   1.24398553e+00]	[ -1.48149023e-02   9.10000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    195	76    	[ -1.31376292e-02   9.55600000e+01]	[  3.20253586e-04   1.64511398e+00]	[ -1.50083956e-02   9.30000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    196	75    	[ -1.31372724e-02   9.52900000e+01]	[  2.77667736e-04   1.20245582e+00]	[ -1.42089605e-02   9.20000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    197	75    	[ -1.31822493e-02   9.51200000e+01]	[  4.36159551e-04   1.57022291e+00]	[ -1.58771069e-02   9.00000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    198	84    	[ -1.31245887e-02   9.53500000e+01]	[  3.12079630e-04   1.55161206e+00]	[ -1.50023272e-02   9.00000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    199	73    	[ -1.31671042e-02   9.54100000e+01]	[  6.34249597e-04   1.54334053e+00]	[ -1.90122288e-02   9.30000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    200	79    	[ -1.31276520e-02   9.52800000e+01]	[  2.94781034e-04   1.44277510e+00]	[ -1.48779874e-02   9.10000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    201	75    	[ -1.31573434e-02   9.51900000e+01]	[  3.59037469e-04   2.04301248e+00]	[ -1.50783993e-02   8.80000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    202	71    	[ -1.31451533e-02   9.54300000e+01]	[  3.41177869e-04   1.63251340e+00]	[ -1.50335239e-02   9.10000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    203	82    	[ -1.31992319e-02   9.54400000e+01]	[  4.36194756e-04   2.01653168e+00]	[ -1.54689084e-02   8.90000000e+01]	[ -1.30216692e-02   1.04000000e+02]
    204	66    	[ -1.30853047e-02   9.53800000e+01]	[  1.86577397e-04   1.91718544e+00]	[ -1.39365431e-02   9.00000000e+01]	[ -1.30216692e-02   1.05000000e+02]
    205	74    	[ -1.30962230e-02   9.56300000e+01]	[  2.24391354e-04   1.78131974e+00]	[ -1.44347957e-02   9.20000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    206	85    	[ -1.30752375e-02   9.52000000e+01]	[  1.97638196e-04   1.43527001e+00]	[ -1.46555021e-02   8.80000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    207	74    	[ -1.31298244e-02   9.52400000e+01]	[  3.22466762e-04   1.55640612e+00]	[ -1.46845917e-02   9.00000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    208	69    	[ -1.31154658e-02   9.53400000e+01]	[  2.96897736e-04   1.62000000e+00]	[ -1.50690803e-02   9.00000000e+01]	[ -1.30216692e-02   1.05000000e+02]
    209	81    	[ -1.31737225e-02   9.53700000e+01]	[  6.74584632e-04   1.58527600e+00]	[ -1.92732341e-02   9.10000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    210	80    	[ -1.31294865e-02   9.52600000e+01]	[  2.67788662e-04   1.57238672e+00]	[ -1.43722250e-02   9.10000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    211	78    	[ -1.31964063e-02   9.53600000e+01]	[  4.40066952e-04   1.21260051e+00]	[ -1.57490622e-02   9.30000000e+01]	[ -1.30216692e-02   1.00000000e+02]
    212	80    	[ -1.31494522e-02   9.52200000e+01]	[  6.29015641e-04   1.43930539e+00]	[ -1.91169469e-02   8.90000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    213	74    	[ -1.31046693e-02   9.55800000e+01]	[  2.37708024e-04   2.14560015e+00]	[ -1.44301978e-02   8.90000000e+01]	[ -1.30216692e-02   1.05000000e+02]
    214	75    	[ -1.31617960e-02   9.53400000e+01]	[  6.52214200e-04   1.26664912e+00]	[ -1.89361004e-02   9.30000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    215	74    	[ -1.31109365e-02   9.51400000e+01]	[  2.72674110e-04   1.43540935e+00]	[ -1.46036405e-02   8.90000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    216	80    	[ -1.31122781e-02   9.54700000e+01]	[  2.60554689e-04   1.96700280e+00]	[ -1.48216661e-02   8.80000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    217	75    	[ -1.31685648e-02   9.52500000e+01]	[  3.47322716e-04   1.76847392e+00]	[ -1.47345392e-02   9.00000000e+01]	[ -1.30216692e-02   1.03000000e+02]
    218	78    	[ -1.31629859e-02   9.51300000e+01]	[  4.31034689e-04   1.27792801e+00]	[ -1.57957709e-02   9.20000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    219	81    	[ -1.32217931e-02   9.54800000e+01]	[  7.35264972e-04   1.62160414e+00]	[ -1.93857199e-02   8.90000000e+01]	[ -1.30216692e-02   1.02000000e+02]
    220	76    	[ -1.31301411e-02   9.51700000e+01]	[  3.12277724e-04   1.49703039e+00]	[ -1.47282463e-02   8.80000000e+01]	[ -1.30216692e-02   1.01000000e+02]
    221	70    	[ -1.30983887e-02   9.51900000e+01]	[  2.68941167e-04   1.30149914e+00]	[ -1.48618064e-02   9.00000000e+01]	[ -1.30214992e-02   1.03000000e+02]
    222	78    	[ -1.31314832e-02   9.51000000e+01]	[  3.09712725e-04   1.26885775e+00]	[ -1.45076327e-02   8.90000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    223	72    	[ -1.31619355e-02   9.56100000e+01]	[  5.84283898e-04   1.88623965e+00]	[ -1.83462543e-02   8.90000000e+01]	[ -1.30200378e-02   1.04000000e+02]
    224	70    	[ -1.31245368e-02   9.58300000e+01]	[  2.97593638e-04   1.57515079e+00]	[ -1.45833352e-02   9.10000000e+01]	[ -1.30200378e-02   1.01000000e+02]
    225	69    	[ -1.30991843e-02   9.61300000e+01]	[  2.79865967e-04   2.11023695e+00]	[ -1.52325131e-02   9.20000000e+01]	[ -1.30200378e-02   1.07000000e+02]
    226	55    	[ -1.31284629e-02   9.56600000e+01]	[  3.11185688e-04   1.47797158e+00]	[ -1.46216115e-02   9.20000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    227	76    	[ -1.31210810e-02   9.60200000e+01]	[  2.52705238e-04   2.02474690e+00]	[ -1.45231083e-02   9.10000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    228	73    	[ -1.30833784e-02   9.61500000e+01]	[  1.87132716e-04   2.03162497e+00]	[ -1.40032643e-02   9.00000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    229	81    	[ -1.31602719e-02   9.58800000e+01]	[  3.26212672e-04   1.55743379e+00]	[ -1.46279664e-02   9.20000000e+01]	[ -1.30200378e-02   1.05000000e+02]
    230	70    	[ -1.30747577e-02   9.56500000e+01]	[  1.92818467e-04   1.36656504e+00]	[ -1.45193818e-02   9.10000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    231	79    	[ -1.31143902e-02   9.57400000e+01]	[  3.30762783e-04   1.79231694e+00]	[ -1.54808817e-02   9.40000000e+01]	[ -1.30200378e-02   1.04000000e+02]
    232	68    	[ -1.31191207e-02   9.51200000e+01]	[  2.68210695e-04   1.65093913e+00]	[ -1.47256345e-02   8.90000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    233	70    	[ -1.30731921e-02   9.50500000e+01]	[  1.74734045e-04   1.34443297e+00]	[ -1.45144462e-02   9.00000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    234	83    	[ -1.30859916e-02   9.52800000e+01]	[  1.97738390e-04   1.07777549e+00]	[ -1.41765956e-02   9.20000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    235	64    	[ -1.30782450e-02   9.53600000e+01]	[  1.94892843e-04   1.51340675e+00]	[ -1.42985481e-02   9.20000000e+01]	[ -1.30200378e-02   1.04000000e+02]
    236	65    	[ -1.31118355e-02   9.50600000e+01]	[  2.84397646e-04   1.52197240e+00]	[ -1.44797449e-02   8.90000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    237	74    	[ -1.31784873e-02   9.51200000e+01]	[  7.88414843e-04   1.15134704e+00]	[ -2.06816105e-02   9.10000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    238	80    	[ -1.31547382e-02   9.54100000e+01]	[  3.79691431e-04   1.49729757e+00]	[ -1.56260461e-02   9.10000000e+01]	[ -1.30200378e-02   1.01000000e+02]
    239	79    	[ -1.31596888e-02   9.52800000e+01]	[  4.23472441e-04   1.80599003e+00]	[ -1.61396450e-02   9.00000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    240	80    	[ -1.31223999e-02   9.53300000e+01]	[  3.00841032e-04   1.02034308e+00]	[ -1.51723895e-02   9.30000000e+01]	[ -1.30200378e-02   9.90000000e+01]
    241	74    	[ -1.31338408e-02   9.54300000e+01]	[  3.17368815e-04   1.36568664e+00]	[ -1.48153968e-02   9.30000000e+01]	[ -1.30200378e-02   1.01000000e+02]
    242	82    	[ -1.31938950e-02   9.53100000e+01]	[  4.70805106e-04   1.57920866e+00]	[ -1.54956130e-02   9.10000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    243	77    	[ -1.31691655e-02   9.54300000e+01]	[  4.17657717e-04   1.90396954e+00]	[ -1.56605505e-02   9.20000000e+01]	[ -1.30200378e-02   1.07000000e+02]
    244	80    	[ -1.31331728e-02   9.49000000e+01]	[  3.21766151e-04   1.41774469e+00]	[ -1.47399765e-02   8.80000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    245	84    	[ -1.31653696e-02   9.48700000e+01]	[  3.57825726e-04   1.56623753e+00]	[ -1.49933749e-02   8.80000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    246	78    	[ -1.31266337e-02   9.46900000e+01]	[  2.73969120e-04   1.29379287e+00]	[ -1.43546359e-02   9.10000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    247	73    	[ -1.32091396e-02   9.46900000e+01]	[  4.25415756e-04   2.07699302e+00]	[ -1.51286949e-02   9.10000000e+01]	[ -1.30200378e-02   1.04000000e+02]
    248	83    	[ -1.31241726e-02   9.44000000e+01]	[  3.08061620e-04   1.40000000e+00]	[ -1.47583508e-02   8.80000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    249	83    	[ -1.31109770e-02   9.42700000e+01]	[  3.18855846e-04   1.21536003e+00]	[ -1.48300711e-02   9.10000000e+01]	[ -1.30200378e-02   1.01000000e+02]
    250	75    	[ -1.30978148e-02   9.43900000e+01]	[  2.34985725e-04   1.49596123e+00]	[ -1.45942362e-02   9.00000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    251	74    	[ -1.31253273e-02   9.41900000e+01]	[  3.52396714e-04   1.36157996e+00]	[ -1.57560950e-02   9.00000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    252	70    	[ -1.30920966e-02   9.42800000e+01]	[  2.23625377e-04   1.58795466e+00]	[ -1.43482088e-02   9.00000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    253	77    	[ -1.31058084e-02   9.40900000e+01]	[  2.90920989e-04   1.48388005e+00]	[ -1.47967557e-02   8.90000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    254	78    	[ -1.32073525e-02   9.45800000e+01]	[  4.89528481e-04   2.06968597e+00]	[ -1.64474665e-02   8.80000000e+01]	[ -1.30200378e-02   1.04000000e+02]
    255	73    	[ -1.31782542e-02   9.46900000e+01]	[  3.69493078e-04   1.96822255e+00]	[ -1.51360930e-02   9.00000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    256	76    	[ -1.31886046e-02   9.45500000e+01]	[  7.22343629e-04   1.81865335e+00]	[ -1.94657674e-02   9.00000000e+01]	[ -1.30200378e-02   1.05000000e+02]
    257	73    	[ -1.31426895e-02   9.42000000e+01]	[  3.97379543e-04   1.45602198e+00]	[ -1.54824069e-02   9.10000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    258	72    	[ -1.30954884e-02   9.45500000e+01]	[  2.35229246e-04   1.69926455e+00]	[ -1.46584302e-02   9.30000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    259	80    	[ -1.32245283e-02   9.42900000e+01]	[  7.24098782e-04   2.20587851e+00]	[ -1.95294453e-02   8.70000000e+01]	[ -1.30200378e-02   1.04000000e+02]
    260	73    	[ -1.31405880e-02   9.43700000e+01]	[  2.93467212e-04   1.63496177e+00]	[ -1.46648284e-02   8.90000000e+01]	[ -1.30200378e-02   1.01000000e+02]
    261	71    	[ -1.31613548e-02   9.47100000e+01]	[  3.78943633e-04   1.92507143e+00]	[ -1.50219231e-02   9.20000000e+01]	[ -1.30200378e-02   1.04000000e+02]
    262	66    	[ -1.30905733e-02   9.44100000e+01]	[  3.02464922e-04   1.66790287e+00]	[ -1.56079037e-02   8.90000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    263	72    	[ -1.30874296e-02   9.40400000e+01]	[  2.76203507e-04   9.68710483e-01]	[ -1.48456713e-02   9.10000000e+01]	[ -1.30200378e-02   9.90000000e+01]
    264	77    	[ -1.31315531e-02   9.40000000e+01]	[  3.53340409e-04   1.32664992e+00]	[ -1.52511891e-02   9.10000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    265	75    	[ -1.31228161e-02   9.38200000e+01]	[  2.84706534e-04   1.49919979e+00]	[ -1.48093085e-02   9.10000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    266	75    	[ -1.30983355e-02   9.38100000e+01]	[  2.40391535e-04   1.65948787e+00]	[ -1.45934309e-02   9.10000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    267	81    	[ -1.31382799e-02   9.40500000e+01]	[  3.22038411e-04   2.38065117e+00]	[ -1.47038801e-02   9.20000000e+01]	[ -1.30200378e-02   1.05000000e+02]
    268	79    	[ -1.31459101e-02   9.35400000e+01]	[  2.93858376e-04   2.05630737e+00]	[ -1.42872043e-02   8.80000000e+01]	[ -1.30200378e-02   1.06000000e+02]
    269	77    	[ -1.31587000e-02   9.31900000e+01]	[  5.21893362e-04   1.71869136e+00]	[ -1.72469539e-02   8.80000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    270	69    	[ -1.31683721e-02   9.34800000e+01]	[  4.26618347e-04   2.18394139e+00]	[ -1.64732312e-02   8.50000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    271	80    	[ -1.30772720e-02   9.33500000e+01]	[  2.00874116e-04   1.63324830e+00]	[ -1.42784504e-02   8.70000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    272	73    	[ -1.32090270e-02   9.33600000e+01]	[  5.04911848e-04   1.68831277e+00]	[ -1.55306444e-02   8.80000000e+01]	[ -1.30200378e-02   1.01000000e+02]
    273	80    	[ -1.31488981e-02   9.33400000e+01]	[  3.58565076e-04   1.81780087e+00]	[ -1.53554829e-02   8.70000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    274	80    	[ -1.31441660e-02   9.33100000e+01]	[  3.15844738e-04   1.41911945e+00]	[ -1.47710129e-02   9.00000000e+01]	[ -1.30200378e-02   1.01000000e+02]
    275	78    	[ -1.31236063e-02   9.33500000e+01]	[  2.85007000e-04   1.71682847e+00]	[ -1.43839632e-02   8.90000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    276	76    	[ -1.30873393e-02   9.33400000e+01]	[  2.46861398e-04   1.49813217e+00]	[ -1.50394546e-02   8.90000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    277	72    	[ -1.30873744e-02   9.33700000e+01]	[  2.43033283e-04   1.46734454e+00]	[ -1.50633572e-02   8.90000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    278	68    	[ -1.31258762e-02   9.34700000e+01]	[  2.64902507e-04   1.63984755e+00]	[ -1.45350360e-02   8.80000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    279	79    	[ -1.31007362e-02   9.36600000e+01]	[  2.09695685e-04   2.01603571e+00]	[ -1.43494544e-02   8.80000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    280	80    	[ -1.31535310e-02   9.36700000e+01]	[  2.92895491e-04   2.15432124e+00]	[ -1.47268762e-02   8.80000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    281	80    	[ -1.31322341e-02   9.35100000e+01]	[  3.65533329e-04   1.93646585e+00]	[ -1.59617766e-02   8.70000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    282	68    	[ -1.32173264e-02   9.36200000e+01]	[  4.53502723e-04   1.96865436e+00]	[ -1.54830979e-02   8.80000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    283	75    	[ -1.32451074e-02   9.30800000e+01]	[  1.09665887e-03   1.96305884e+00]	[ -2.36100721e-02   8.40000000e+01]	[ -1.30200378e-02   9.90000000e+01]
    284	65    	[ -1.31090381e-02   9.32300000e+01]	[  2.51101561e-04   8.92804570e-01]	[ -1.46044254e-02   9.00000000e+01]	[ -1.30200378e-02   9.80000000e+01]
    285	71    	[ -1.31440443e-02   9.35500000e+01]	[  3.78894390e-04   1.58350876e+00]	[ -1.56979675e-02   9.10000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    286	87    	[ -1.31362791e-02   9.33200000e+01]	[  3.16191532e-04   1.45519758e+00]	[ -1.52012648e-02   8.80000000e+01]	[ -1.30200378e-02   9.90000000e+01]
    287	72    	[ -1.31351832e-02   9.34500000e+01]	[  2.50781865e-04   1.81865335e+00]	[ -1.41331063e-02   8.90000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    288	69    	[ -1.30959950e-02   9.35900000e+01]	[  2.82247227e-04   2.09807054e+00]	[ -1.54484040e-02   9.00000000e+01]	[ -1.30200378e-02   1.07000000e+02]
    289	78    	[ -1.31062691e-02   9.32900000e+01]	[  2.77832320e-04   1.34383779e+00]	[ -1.48509173e-02   8.70000000e+01]	[ -1.30200378e-02   9.90000000e+01]
    290	76    	[ -1.31437107e-02   9.33900000e+01]	[  2.91164500e-04   2.12082531e+00]	[ -1.43260032e-02   8.70000000e+01]	[ -1.30200378e-02   1.01000000e+02]
    291	78    	[ -1.31469264e-02   9.32400000e+01]	[  3.48001405e-04   1.42211111e+00]	[ -1.48845114e-02   8.80000000e+01]	[ -1.30200378e-02   9.90000000e+01]
    292	80    	[ -1.31427699e-02   9.33700000e+01]	[  3.29310660e-04   1.61031053e+00]	[ -1.46427204e-02   8.90000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    293	77    	[ -1.31980388e-02   9.32200000e+01]	[  4.85063989e-04   2.02277038e+00]	[ -1.65108037e-02   8.70000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    294	76    	[ -1.31044783e-02   9.32800000e+01]	[  2.63227980e-04   1.05905618e+00]	[ -1.46393717e-02   9.10000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    295	76    	[ -1.31302479e-02   9.36200000e+01]	[  2.93670640e-04   1.91196234e+00]	[ -1.50797043e-02   9.00000000e+01]	[ -1.30200378e-02   1.03000000e+02]
    296	65    	[ -1.30987676e-02   9.33800000e+01]	[  1.91828141e-04   1.64182825e+00]	[ -1.41390608e-02   9.00000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    297	70    	[ -1.30965988e-02   9.33700000e+01]	[  2.43618445e-04   1.62883394e+00]	[ -1.48636357e-02   8.90000000e+01]	[ -1.30200378e-02   1.01000000e+02]
    298	79    	[ -1.30960657e-02   9.32700000e+01]	[  2.69802696e-04   1.45502577e+00]	[ -1.51608781e-02   8.80000000e+01]	[ -1.30200378e-02   1.00000000e+02]
    299	81    	[ -1.30901575e-02   9.31700000e+01]	[  2.28183081e-04   1.12298709e+00]	[ -1.42950956e-02   8.90000000e+01]	[ -1.30200378e-02   9.80000000e+01]
    300	77    	[ -1.30809792e-02   9.34700000e+01]	[  2.01950845e-04   1.69973527e+00]	[ -1.46439658e-02   9.10000000e+01]	[ -1.30200378e-02   1.02000000e+02]
    Selecting features with genetic algorithm.
    gen	nevals	avg                                	std                                	min                                	max                                
    0  	100   	[ -1.86853367e-02   1.09460000e+02]	[  2.29391516e-03   6.65795764e+00]	[ -2.59124895e-02   9.50000000e+01]	[ -1.41401062e-02   1.24000000e+02]
    1  	84    	[ -1.72121130e-02   1.10070000e+02]	[  1.77677084e-03   6.55782738e+00]	[ -2.18805334e-02   9.50000000e+01]	[ -1.42056119e-02   1.24000000e+02]
    2  	78    	[ -1.61349348e-02   1.09800000e+02]	[  1.37116106e-03   6.36396103e+00]	[ -2.12893898e-02   9.50000000e+01]	[ -1.41163135e-02   1.25000000e+02]
    3  	65    	[ -1.53656484e-02   1.10560000e+02]	[  8.86876665e-04   5.51782566e+00]	[ -1.81945924e-02   9.50000000e+01]	[ -1.39960280e-02   1.27000000e+02]
    4  	74    	[ -1.48189413e-02   1.11430000e+02]	[  6.05200201e-04   6.07495679e+00]	[ -1.70120063e-02   9.70000000e+01]	[ -1.37034462e-02   1.26000000e+02]
    5  	82    	[ -1.44759528e-02   1.12380000e+02]	[  5.83307161e-04   6.84949633e+00]	[ -1.72202102e-02   9.70000000e+01]	[ -1.36315727e-02   1.27000000e+02]
    6  	78    	[ -1.42062211e-02   1.14250000e+02]	[  4.62750592e-04   6.08009046e+00]	[ -1.59547665e-02   1.02000000e+02]	[ -1.32362908e-02   1.29000000e+02]
    7  	82    	[ -1.40975965e-02   1.14430000e+02]	[  4.55567278e-04   6.22616254e+00]	[ -1.59678414e-02   1.01000000e+02]	[ -1.31933819e-02   1.30000000e+02]
    8  	78    	[ -1.38729400e-02   1.14260000e+02]	[  4.00867747e-04   5.89681270e+00]	[ -1.50909642e-02   1.02000000e+02]	[ -1.31933819e-02   1.30000000e+02]
    9  	76    	[ -1.37518501e-02   1.14780000e+02]	[  3.68824237e-04   6.44450153e+00]	[ -1.53978076e-02   9.80000000e+01]	[ -1.30682149e-02   1.30000000e+02]
    10 	78    	[ -1.36421322e-02   1.14360000e+02]	[  3.17163572e-04   6.82131952e+00]	[ -1.45225820e-02   1.01000000e+02]	[ -1.29708934e-02   1.27000000e+02]
    11 	89    	[ -1.35689717e-02   1.13500000e+02]	[  3.02509835e-04   6.12127438e+00]	[ -1.43890784e-02   1.00000000e+02]	[ -1.29453723e-02   1.29000000e+02]
    12 	84    	[ -1.35200135e-02   1.13030000e+02]	[  4.18343204e-04   6.17163674e+00]	[ -1.53829815e-02   9.60000000e+01]	[ -1.29453723e-02   1.25000000e+02]
    13 	69    	[ -1.34077131e-02   1.13100000e+02]	[  3.17440923e-04   6.94046108e+00]	[ -1.48549763e-02   9.60000000e+01]	[ -1.27109801e-02   1.26000000e+02]
    14 	72    	[ -1.33563022e-02   1.12490000e+02]	[  4.08073148e-04   7.62691943e+00]	[ -1.59694169e-02   9.50000000e+01]	[ -1.27109801e-02   1.29000000e+02]
    15 	72    	[ -1.32293539e-02   1.13260000e+02]	[  3.18755908e-04   6.27792960e+00]	[ -1.49751647e-02   9.50000000e+01]	[ -1.27109801e-02   1.30000000e+02]
    16 	77    	[ -1.31620404e-02   1.13300000e+02]	[  2.98651435e-04   5.32822672e+00]	[ -1.42587514e-02   9.90000000e+01]	[ -1.26940141e-02   1.25000000e+02]
    17 	78    	[ -1.30976996e-02   1.13740000e+02]	[  4.52229895e-04   4.57519398e+00]	[ -1.53121479e-02   1.02000000e+02]	[ -1.25426148e-02   1.28000000e+02]
    18 	77    	[ -1.29922888e-02   1.13510000e+02]	[  2.36797594e-04   4.22017772e+00]	[ -1.37445193e-02   1.03000000e+02]	[ -1.25426148e-02   1.28000000e+02]
    19 	86    	[ -1.30991251e-02   1.12380000e+02]	[  4.20723413e-04   4.19232632e+00]	[ -1.57833011e-02   9.80000000e+01]	[ -1.25048982e-02   1.24000000e+02]
    20 	77    	[ -1.29822938e-02   1.13140000e+02]	[  2.78124569e-04   3.87819546e+00]	[ -1.41046347e-02   1.01000000e+02]	[ -1.23714993e-02   1.24000000e+02]
    21 	83    	[ -1.29862271e-02   1.12570000e+02]	[  3.54623357e-04   3.76896538e+00]	[ -1.44265828e-02   1.03000000e+02]	[ -1.23714993e-02   1.23000000e+02]
    22 	75    	[ -1.29699320e-02   1.13080000e+02]	[  2.96850406e-04   3.48620137e+00]	[ -1.44898125e-02   1.06000000e+02]	[ -1.25279905e-02   1.21000000e+02]
    23 	87    	[ -1.30214904e-02   1.12960000e+02]	[  4.11289770e-04   3.73609422e+00]	[ -1.50099635e-02   1.04000000e+02]	[ -1.24910542e-02   1.22000000e+02]
    24 	75    	[ -1.29095259e-02   1.12820000e+02]	[  2.48255966e-04   3.62044196e+00]	[ -1.41997007e-02   1.04000000e+02]	[ -1.24004049e-02   1.23000000e+02]
    25 	73    	[ -1.29298902e-02   1.12330000e+02]	[  3.34338375e-04   3.47866354e+00]	[ -1.50229933e-02   1.04000000e+02]	[ -1.24004049e-02   1.23000000e+02]
    26 	81    	[ -1.29330064e-02   1.12330000e+02]	[  3.00703875e-04   3.31980421e+00]	[ -1.40848888e-02   1.01000000e+02]	[ -1.24645270e-02   1.23000000e+02]
    27 	61    	[ -1.29404484e-02   1.11540000e+02]	[  3.86463650e-04   3.63158368e+00]	[ -1.46236368e-02   1.01000000e+02]	[ -1.24645270e-02   1.20000000e+02]
    28 	76    	[ -1.28979997e-02   1.11280000e+02]	[  3.61244336e-04   3.43534569e+00]	[ -1.51826190e-02   1.01000000e+02]	[ -1.25028455e-02   1.18000000e+02]
    29 	71    	[ -1.28615786e-02   1.10920000e+02]	[  3.49421835e-04   3.78333186e+00]	[ -1.54722984e-02   9.90000000e+01]	[ -1.24303831e-02   1.21000000e+02]
    30 	72    	[ -1.28153143e-02   1.10810000e+02]	[  2.62549985e-04   3.00231577e+00]	[ -1.39417167e-02   1.04000000e+02]	[ -1.24303831e-02   1.18000000e+02]
    31 	81    	[ -1.28984979e-02   1.10210000e+02]	[  4.17849899e-04   3.70484818e+00]	[ -1.53775547e-02   9.90000000e+01]	[ -1.24622685e-02   1.19000000e+02]
    32 	70    	[ -1.28251570e-02   1.10610000e+02]	[  2.78071243e-04   3.13015974e+00]	[ -1.43289809e-02   1.04000000e+02]	[ -1.24837044e-02   1.19000000e+02]
    33 	71    	[ -1.29242986e-02   1.10390000e+02]	[  5.18032845e-04   3.71993280e+00]	[ -1.57199907e-02   1.03000000e+02]	[ -1.24837044e-02   1.20000000e+02]
    34 	90    	[ -1.28740129e-02   1.09690000e+02]	[  3.23539127e-04   3.37844639e+00]	[ -1.46403578e-02   1.01000000e+02]	[ -1.24366587e-02   1.18000000e+02]
    35 	69    	[ -1.28496823e-02   1.09880000e+02]	[  3.02753310e-04   3.87886581e+00]	[ -1.42442138e-02   1.00000000e+02]	[ -1.23649080e-02   1.19000000e+02]
    36 	80    	[ -1.27759846e-02   1.09500000e+02]	[  2.22323011e-04   3.22955105e+00]	[ -1.34748049e-02   1.03000000e+02]	[ -1.23649080e-02   1.18000000e+02]
    37 	80    	[ -1.28353967e-02   1.09470000e+02]	[  3.80180151e-04   3.17003155e+00]	[ -1.48819550e-02   1.02000000e+02]	[ -1.24176051e-02   1.19000000e+02]
    38 	68    	[ -1.27623744e-02   1.08400000e+02]	[  2.98237286e-04   3.43511281e+00]	[ -1.39810027e-02   1.01000000e+02]	[ -1.22315217e-02   1.22000000e+02]
    39 	73    	[ -1.27716487e-02   1.07830000e+02]	[  3.83974772e-04   2.51019920e+00]	[ -1.52581593e-02   1.00000000e+02]	[ -1.22315217e-02   1.16000000e+02]
    40 	76    	[ -1.27452321e-02   1.08080000e+02]	[  3.43215978e-04   2.77373395e+00]	[ -1.47484525e-02   1.01000000e+02]	[ -1.22785963e-02   1.18000000e+02]
    41 	78    	[ -1.27838538e-02   1.07510000e+02]	[  3.74538691e-04   2.41451859e+00]	[ -1.53533349e-02   9.90000000e+01]	[ -1.22785963e-02   1.16000000e+02]
    42 	81    	[ -1.27663653e-02   1.07710000e+02]	[  2.64136584e-04   2.40538978e+00]	[ -1.40659091e-02   1.01000000e+02]	[ -1.22785963e-02   1.14000000e+02]
    43 	81    	[ -1.28192753e-02   1.07300000e+02]	[  3.36202111e-04   2.46779254e+00]	[ -1.45137186e-02   1.01000000e+02]	[ -1.23954364e-02   1.15000000e+02]
    44 	75    	[ -1.28522840e-02   1.07500000e+02]	[  3.86624365e-04   2.98831056e+00]	[ -1.47886284e-02   9.70000000e+01]	[ -1.24050037e-02   1.15000000e+02]
    45 	73    	[ -1.27381964e-02   1.07840000e+02]	[  2.17115841e-04   2.80257025e+00]	[ -1.38507776e-02   1.00000000e+02]	[ -1.23744795e-02   1.15000000e+02]
    46 	71    	[ -1.27613005e-02   1.08220000e+02]	[  2.81932647e-04   3.07759646e+00]	[ -1.43652196e-02   1.01000000e+02]	[ -1.23601443e-02   1.17000000e+02]
    47 	86    	[ -1.28096027e-02   1.07960000e+02]	[  4.43473039e-04   2.53740024e+00]	[ -1.52530242e-02   1.02000000e+02]	[ -1.22857992e-02   1.14000000e+02]
    48 	78    	[ -1.27925124e-02   1.07860000e+02]	[  3.95800041e-04   2.66465758e+00]	[ -1.50562732e-02   1.00000000e+02]	[ -1.22857992e-02   1.16000000e+02]
    49 	69    	[ -1.27891492e-02   1.07470000e+02]	[  4.48819201e-04   2.75845972e+00]	[ -1.51395569e-02   1.00000000e+02]	[ -1.23409115e-02   1.17000000e+02]
    50 	77    	[ -1.28079974e-02   1.07440000e+02]	[  4.16447685e-04   2.78682615e+00]	[ -1.54060232e-02   9.90000000e+01]	[ -1.23174915e-02   1.14000000e+02]
    51 	73    	[ -1.27627798e-02   1.07520000e+02]	[  3.75564636e-04   2.91369182e+00]	[ -1.52511355e-02   1.03000000e+02]	[ -1.23174915e-02   1.17000000e+02]
    52 	75    	[ -1.27083465e-02   1.07010000e+02]	[  2.54887432e-04   2.27813520e+00]	[ -1.42019157e-02   1.03000000e+02]	[ -1.23174915e-02   1.13000000e+02]
    53 	64    	[ -1.27356036e-02   1.06900000e+02]	[  4.01277644e-04   2.66645833e+00]	[ -1.49014155e-02   9.90000000e+01]	[ -1.23174915e-02   1.15000000e+02]
    54 	83    	[ -1.27150845e-02   1.07120000e+02]	[  3.46397488e-04   2.83647669e+00]	[ -1.44434044e-02   1.01000000e+02]	[ -1.23174915e-02   1.18000000e+02]
    55 	75    	[ -1.26778388e-02   1.07350000e+02]	[  3.05343949e-04   2.64338798e+00]	[ -1.44330126e-02   9.90000000e+01]	[ -1.23042404e-02   1.15000000e+02]
    56 	79    	[ -1.26869610e-02   1.07560000e+02]	[  3.51250093e-04   2.17862342e+00]	[ -1.46553463e-02   1.00000000e+02]	[ -1.23042404e-02   1.14000000e+02]
    57 	64    	[ -1.26477397e-02   1.08030000e+02]	[  3.25851527e-04   2.08064894e+00]	[ -1.42649345e-02   1.03000000e+02]	[ -1.21548436e-02   1.13000000e+02]
    58 	72    	[ -1.26758294e-02   1.08880000e+02]	[  3.30878791e-04   2.58178233e+00]	[ -1.44166797e-02   1.02000000e+02]	[ -1.21548436e-02   1.19000000e+02]
    59 	77    	[ -1.26721598e-02   1.08720000e+02]	[  3.16061276e-04   2.51825336e+00]	[ -1.37179122e-02   1.00000000e+02]	[ -1.21548436e-02   1.16000000e+02]
    60 	86    	[ -1.27021416e-02   1.08890000e+02]	[  4.24953582e-04   1.78266654e+00]	[ -1.48254871e-02   1.03000000e+02]	[ -1.22447331e-02   1.14000000e+02]
    61 	73    	[ -1.26356465e-02   1.08890000e+02]	[  4.92399081e-04   1.63642904e+00]	[ -1.59274047e-02   1.01000000e+02]	[ -1.22447331e-02   1.12000000e+02]
    62 	86    	[ -1.26372900e-02   1.09120000e+02]	[  5.09388824e-04   2.21485891e+00]	[ -1.49867894e-02   1.01000000e+02]	[ -1.22447331e-02   1.16000000e+02]
    63 	75    	[ -1.25290966e-02   1.09080000e+02]	[  3.32173041e-04   1.76453960e+00]	[ -1.40293601e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    64 	72    	[ -1.24730963e-02   1.09210000e+02]	[  3.33377935e-04   1.25135926e+00]	[ -1.37560989e-02   1.04000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    65 	74    	[ -1.25057604e-02   1.08820000e+02]	[  5.22802190e-04   1.10797112e+00]	[ -1.65196740e-02   1.03000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    66 	70    	[ -1.25021301e-02   1.08800000e+02]	[  3.23495300e-04   1.94422221e+00]	[ -1.37166472e-02   9.90000000e+01]	[ -1.23358617e-02   1.14000000e+02]
    67 	70    	[ -1.24542747e-02   1.08810000e+02]	[  2.51514327e-04   1.48118196e+00]	[ -1.32907478e-02   1.03000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    68 	66    	[ -1.24521900e-02   1.09090000e+02]	[  2.96791170e-04   1.28136646e+00]	[ -1.43597156e-02   1.06000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    69 	81    	[ -1.24412811e-02   1.09080000e+02]	[  3.25694169e-04   1.11964280e+00]	[ -1.45427031e-02   1.04000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    70 	80    	[ -1.24786763e-02   1.09130000e+02]	[  3.72356868e-04   1.61031053e+00]	[ -1.44176952e-02   1.05000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    71 	75    	[ -1.25167922e-02   1.09230000e+02]	[  4.66074668e-04   1.69620164e+00]	[ -1.52506177e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    72 	79    	[ -1.25117548e-02   1.09100000e+02]	[  3.96893031e-04   1.90000000e+00]	[ -1.50771803e-02   1.00000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    73 	71    	[ -1.25355258e-02   1.09450000e+02]	[  4.18635363e-04   1.75142799e+00]	[ -1.44527373e-02   1.06000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    74 	71    	[ -1.25137889e-02   1.08890000e+02]	[  4.01361469e-04   1.87560657e+00]	[ -1.42252993e-02   1.02000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    75 	83    	[ -1.24913100e-02   1.09030000e+02]	[  4.10576646e-04   1.21206435e+00]	[ -1.44645018e-02   1.04000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    76 	77    	[ -1.24218857e-02   1.09000000e+02]	[  2.68431812e-04   1.23288280e+00]	[ -1.39377343e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    77 	80    	[ -1.25564442e-02   1.09000000e+02]	[  5.23689589e-04   1.23288280e+00]	[ -1.51057857e-02   1.04000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    78 	75    	[ -1.25011976e-02   1.08910000e+02]	[  3.55800370e-04   1.44979309e+00]	[ -1.42307450e-02   1.03000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    79 	75    	[ -1.24753743e-02   1.08860000e+02]	[  3.70143346e-04   1.28078101e+00]	[ -1.46329171e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    80 	88    	[ -1.24891811e-02   1.09020000e+02]	[  3.87839454e-04   1.11337325e+00]	[ -1.51021204e-02   1.05000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    81 	75    	[ -1.25005874e-02   1.09480000e+02]	[  3.65866578e-04   1.97220689e+00]	[ -1.44101257e-02   1.04000000e+02]	[ -1.23358617e-02   1.19000000e+02]
    82 	75    	[ -1.24882660e-02   1.09200000e+02]	[  4.55529026e-04   1.37840488e+00]	[ -1.54265081e-02   1.05000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    83 	81    	[ -1.25436511e-02   1.09030000e+02]	[  5.46846373e-04   1.38892044e+00]	[ -1.65827279e-02   1.02000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    84 	72    	[ -1.24604913e-02   1.08830000e+02]	[  3.01230480e-04   1.61279261e+00]	[ -1.40898947e-02   9.80000000e+01]	[ -1.23358617e-02   1.15000000e+02]
    85 	82    	[ -1.24104712e-02   1.09140000e+02]	[  2.56775659e-04   7.48598691e-01]	[ -1.43368744e-02   1.07000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    86 	72    	[ -1.24225205e-02   1.09050000e+02]	[  2.54175441e-04   1.02347447e+00]	[ -1.38003840e-02   1.05000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    87 	77    	[ -1.25381922e-02   1.08960000e+02]	[  5.54458009e-04   1.24835892e+00]	[ -1.56997033e-02   1.05000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    88 	75    	[ -1.25057481e-02   1.08910000e+02]	[  4.21685623e-04   1.29688087e+00]	[ -1.50564285e-02   1.03000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    89 	81    	[ -1.25536982e-02   1.09170000e+02]	[  4.65277639e-04   1.64957570e+00]	[ -1.45671983e-02   1.03000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    90 	62    	[ -1.24582100e-02   1.09190000e+02]	[  3.02715840e-04   1.24655525e+00]	[ -1.41769361e-02   1.04000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    91 	91    	[ -1.25536259e-02   1.08740000e+02]	[  5.14349041e-04   1.57873367e+00]	[ -1.53606276e-02   1.01000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    92 	81    	[ -1.24796881e-02   1.08960000e+02]	[  3.47045809e-04   1.52262930e+00]	[ -1.45547999e-02   1.03000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    93 	75    	[ -1.25231829e-02   1.09150000e+02]	[  3.79987352e-04   1.65151446e+00]	[ -1.46832691e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    94 	68    	[ -1.25559009e-02   1.09240000e+02]	[  4.95458392e-04   1.60698475e+00]	[ -1.47212744e-02   1.03000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    95 	76    	[ -1.24907714e-02   1.08940000e+02]	[  3.16008752e-04   2.00409581e+00]	[ -1.40056108e-02   1.01000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    96 	73    	[ -1.24855086e-02   1.09100000e+02]	[  4.14196237e-04   1.17046999e+00]	[ -1.53697213e-02   1.05000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    97 	69    	[ -1.24980609e-02   1.08870000e+02]	[  3.81766936e-04   1.12831733e+00]	[ -1.46524341e-02   1.04000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    98 	71    	[ -1.25267749e-02   1.09080000e+02]	[  4.44287829e-04   1.21391927e+00]	[ -1.46651037e-02   1.04000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    99 	83    	[ -1.25259064e-02   1.09060000e+02]	[  4.83194834e-04   1.25554769e+00]	[ -1.48547524e-02   1.03000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    100	74    	[ -1.25484978e-02   1.09180000e+02]	[  4.65015268e-04   2.25557088e+00]	[ -1.47619967e-02   1.02000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    101	72    	[ -1.25592413e-02   1.09280000e+02]	[  4.36922119e-04   1.95489130e+00]	[ -1.47737685e-02   1.04000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    102	75    	[ -1.25193798e-02   1.08980000e+02]	[  3.32596494e-04   1.82197695e+00]	[ -1.37122700e-02   1.01000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    103	78    	[ -1.24634024e-02   1.09130000e+02]	[  2.93834029e-04   1.01641527e+00]	[ -1.36984298e-02   1.06000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    104	77    	[ -1.24757263e-02   1.09160000e+02]	[  3.52414703e-04   1.92208220e+00]	[ -1.44614881e-02   1.01000000e+02]	[ -1.23358617e-02   1.19000000e+02]
    105	78    	[ -1.24600833e-02   1.08800000e+02]	[  3.18906014e-04   8.60232527e-01]	[ -1.42463818e-02   1.05000000e+02]	[ -1.23358617e-02   1.11000000e+02]
    106	76    	[ -1.25440554e-02   1.08580000e+02]	[  4.19108161e-04   1.81207064e+00]	[ -1.44350177e-02   9.80000000e+01]	[ -1.23358617e-02   1.14000000e+02]
    107	81    	[ -1.25991337e-02   1.09260000e+02]	[  5.55070918e-04   1.62246726e+00]	[ -1.47814662e-02   1.05000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    108	71    	[ -1.25544345e-02   1.09130000e+02]	[  4.30665915e-04   1.36128616e+00]	[ -1.48575919e-02   1.02000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    109	74    	[ -1.25197798e-02   1.09010000e+02]	[  4.18314868e-04   1.80828648e+00]	[ -1.49421098e-02   1.01000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    110	77    	[ -1.24614436e-02   1.08890000e+02]	[  2.86838047e-04   1.41347091e+00]	[ -1.37607285e-02   1.03000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    111	79    	[ -1.25532557e-02   1.09270000e+02]	[  4.43982325e-04   1.63618459e+00]	[ -1.47523296e-02   1.05000000e+02]	[ -1.23358617e-02   1.19000000e+02]
    112	81    	[ -1.25711245e-02   1.09160000e+02]	[  4.35013840e-04   1.57936696e+00]	[ -1.49740432e-02   1.03000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    113	79    	[ -1.25250832e-02   1.09040000e+02]	[  3.94304497e-04   1.48270024e+00]	[ -1.39754874e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    114	82    	[ -1.25064318e-02   1.08930000e+02]	[  3.79138348e-04   1.19377552e+00]	[ -1.43682742e-02   1.05000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    115	80    	[ -1.24559740e-02   1.08840000e+02]	[  3.02534913e-04   1.07443008e+00]	[ -1.38990516e-02   1.05000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    116	76    	[ -1.24514677e-02   1.08970000e+02]	[  2.91752186e-04   1.38170185e+00]	[ -1.38085114e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    117	76    	[ -1.24580466e-02   1.09040000e+02]	[  3.30205644e-04   9.89141042e-01]	[ -1.45808423e-02   1.04000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    118	83    	[ -1.25013429e-02   1.08960000e+02]	[  4.69450612e-04   8.35703297e-01]	[ -1.60817505e-02   1.05000000e+02]	[ -1.23358617e-02   1.11000000e+02]
    119	72    	[ -1.25090294e-02   1.08940000e+02]	[  4.18812093e-04   1.40584494e+00]	[ -1.45441479e-02   1.01000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    120	81    	[ -1.26271776e-02   1.08740000e+02]	[  5.51988854e-04   1.28545712e+00]	[ -1.50675825e-02   1.03000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    121	79    	[ -1.24720046e-02   1.09120000e+02]	[  3.09995981e-04   1.60174904e+00]	[ -1.41363665e-02   1.03000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    122	83    	[ -1.24794607e-02   1.08930000e+02]	[  4.00989158e-04   1.53137193e+00]	[ -1.46509407e-02   1.02000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    123	82    	[ -1.24548805e-02   1.09090000e+02]	[  2.75688411e-04   1.20909057e+00]	[ -1.34397020e-02   1.04000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    124	80    	[ -1.24779134e-02   1.09290000e+02]	[  3.43754698e-04   1.82918014e+00]	[ -1.42528694e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    125	71    	[ -1.24958920e-02   1.08760000e+02]	[  3.73200227e-04   1.61938260e+00]	[ -1.41571914e-02   1.02000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    126	74    	[ -1.24693620e-02   1.08830000e+02]	[  3.56140581e-04   9.49262872e-01]	[ -1.41694380e-02   1.04000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    127	72    	[ -1.25077339e-02   1.09180000e+02]	[  4.54243287e-04   1.33701159e+00]	[ -1.48937256e-02   1.03000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    128	81    	[ -1.25355344e-02   1.09320000e+02]	[  3.30127524e-04   1.61170717e+00]	[ -1.35940401e-02   1.06000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    129	70    	[ -1.24837138e-02   1.08980000e+02]	[  3.88749655e-04   1.06752049e+00]	[ -1.46918215e-02   1.05000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    130	80    	[ -1.24543752e-02   1.09020000e+02]	[  3.41127919e-04   1.28046866e+00]	[ -1.48939746e-02   1.02000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    131	71    	[ -1.25083319e-02   1.09040000e+02]	[  4.60000117e-04   1.46914941e+00]	[ -1.51790945e-02   1.03000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    132	69    	[ -1.25285314e-02   1.09170000e+02]	[  5.19736337e-04   1.54954832e+00]	[ -1.60742552e-02   1.04000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    133	75    	[ -1.24141667e-02   1.09120000e+02]	[  2.22854168e-04   1.27499020e+00]	[ -1.36184449e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    134	69    	[ -1.24920800e-02   1.08860000e+02]	[  4.47247512e-04   1.27294933e+00]	[ -1.48082390e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    135	76    	[ -1.24725011e-02   1.08970000e+02]	[  4.04036819e-04   7.27392604e-01]	[ -1.46382305e-02   1.06000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    136	80    	[ -1.24708577e-02   1.08830000e+02]	[  3.53492312e-04   1.12298709e+00]	[ -1.40198226e-02   1.03000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    137	75    	[ -1.24923085e-02   1.09130000e+02]	[  3.27209586e-04   1.49435605e+00]	[ -1.41127046e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    138	84    	[ -1.24641093e-02   1.09240000e+02]	[  3.26495290e-04   1.30476051e+00]	[ -1.41836961e-02   1.05000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    139	79    	[ -1.24880533e-02   1.09130000e+02]	[  3.69296301e-04   1.27792801e+00]	[ -1.47423484e-02   1.05000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    140	67    	[ -1.24847818e-02   1.08920000e+02]	[  3.77470987e-04   1.27027556e+00]	[ -1.45482410e-02   1.03000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    141	81    	[ -1.25363169e-02   1.09200000e+02]	[  4.80055743e-04   1.57480157e+00]	[ -1.49177751e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    142	83    	[ -1.24779080e-02   1.09060000e+02]	[  3.45728874e-04   1.50213182e+00]	[ -1.42806259e-02   1.04000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    143	81    	[ -1.24972293e-02   1.09090000e+02]	[  4.33306658e-04   1.78938537e+00]	[ -1.51189082e-02   1.02000000e+02]	[ -1.23358617e-02   1.19000000e+02]
    144	81    	[ -1.24815419e-02   1.08980000e+02]	[  3.84326569e-04   1.03903802e+00]	[ -1.49022547e-02   1.03000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    145	80    	[ -1.25399601e-02   1.08650000e+02]	[  4.06902639e-04   2.13717103e+00]	[ -1.44738207e-02   9.90000000e+01]	[ -1.23358617e-02   1.16000000e+02]
    146	77    	[ -1.24711513e-02   1.08820000e+02]	[  3.18465049e-04   1.12587744e+00]	[ -1.35867431e-02   1.01000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    147	77    	[ -1.25001249e-02   1.09200000e+02]	[  5.25027992e-04   1.37840488e+00]	[ -1.63173228e-02   1.05000000e+02]	[ -1.23358617e-02   1.19000000e+02]
    148	77    	[ -1.24517567e-02   1.08940000e+02]	[  4.23008178e-04   1.39154590e+00]	[ -1.59749324e-02   1.00000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    149	75    	[ -1.25528064e-02   1.09040000e+02]	[  4.23509189e-04   1.61814709e+00]	[ -1.47738641e-02   1.02000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    150	75    	[ -1.24711537e-02   1.09060000e+02]	[  3.19496142e-04   1.52852870e+00]	[ -1.38741804e-02   1.01000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    151	66    	[ -1.24548178e-02   1.08950000e+02]	[  3.18657716e-04   1.26787223e+00]	[ -1.39421167e-02   1.03000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    152	75    	[ -1.24847094e-02   1.08880000e+02]	[  3.09582497e-04   1.15134704e+00]	[ -1.37715611e-02   1.04000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    153	77    	[ -1.24701567e-02   1.09060000e+02]	[  3.71631604e-04   1.23951603e+00]	[ -1.46166484e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    154	80    	[ -1.24969098e-02   1.08970000e+02]	[  3.64858414e-04   1.58401389e+00]	[ -1.41196042e-02   1.01000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    155	81    	[ -1.25162720e-02   1.09020000e+02]	[  4.12327260e-04   1.48310485e+00]	[ -1.51824362e-02   1.03000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    156	73    	[ -1.24629562e-02   1.09230000e+02]	[  3.48012900e-04   1.27949209e+00]	[ -1.44616916e-02   1.04000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    157	81    	[ -1.25704200e-02   1.09080000e+02]	[  4.79433073e-04   1.94771661e+00]	[ -1.48010611e-02   1.00000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    158	76    	[ -1.24717854e-02   1.09210000e+02]	[  3.44729586e-04   1.65707574e+00]	[ -1.47880677e-02   1.03000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    159	62    	[ -1.24873284e-02   1.09300000e+02]	[  3.38696995e-04   1.57797338e+00]	[ -1.46133357e-02   1.04000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    160	72    	[ -1.25065812e-02   1.09220000e+02]	[  4.24715918e-04   1.50053324e+00]	[ -1.52084253e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    161	84    	[ -1.24123074e-02   1.08950000e+02]	[  2.21997145e-04   1.08050914e+00]	[ -1.35367827e-02   1.04000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    162	81    	[ -1.25117786e-02   1.08910000e+02]	[  3.79680326e-04   1.53684742e+00]	[ -1.49228775e-02   1.02000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    163	83    	[ -1.25423286e-02   1.09310000e+02]	[  4.41685514e-04   1.39781973e+00]	[ -1.48980487e-02   1.04000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    164	81    	[ -1.24773853e-02   1.09240000e+02]	[  3.01574703e-04   1.47051012e+00]	[ -1.41273215e-02   1.05000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    165	78    	[ -1.24629027e-02   1.08740000e+02]	[  3.75620193e-04   1.45340978e+00]	[ -1.53611283e-02   1.00000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    166	74    	[ -1.24803251e-02   1.08890000e+02]	[  3.79593045e-04   1.71403034e+00]	[ -1.45931167e-02   1.01000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    167	69    	[ -1.26517629e-02   1.09210000e+02]	[  6.10070884e-04   1.97126863e+00]	[ -1.46360401e-02   1.02000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    168	64    	[ -1.25063075e-02   1.09200000e+02]	[  4.07905935e-04   1.24096736e+00]	[ -1.50309859e-02   1.04000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    169	78    	[ -1.25232530e-02   1.08940000e+02]	[  4.91926126e-04   1.64207186e+00]	[ -1.49192107e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    170	82    	[ -1.24743596e-02   1.09110000e+02]	[  3.64909066e-04   1.06672396e+00]	[ -1.48518147e-02   1.05000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    171	77    	[ -1.25177161e-02   1.09120000e+02]	[  4.49289613e-04   1.72788889e+00]	[ -1.47605633e-02   1.04000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    172	80    	[ -1.24949624e-02   1.09080000e+02]	[  3.56592452e-04   1.45382255e+00]	[ -1.41689375e-02   1.04000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    173	71    	[ -1.24550352e-02   1.08860000e+02]	[  3.24223575e-04   1.22490816e+00]	[ -1.42547214e-02   1.01000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    174	77    	[ -1.25483079e-02   1.09130000e+02]	[  5.57168121e-04   1.67125701e+00]	[ -1.65494017e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    175	76    	[ -1.24902222e-02   1.08760000e+02]	[  4.22538632e-04   9.17823512e-01]	[ -1.51933803e-02   1.04000000e+02]	[ -1.23358617e-02   1.11000000e+02]
    176	78    	[ -1.25077067e-02   1.09320000e+02]	[  4.47839028e-04   1.45519758e+00]	[ -1.50816538e-02   1.05000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    177	77    	[ -1.25049243e-02   1.08760000e+02]	[  3.48969098e-04   1.27373467e+00]	[ -1.39559571e-02   1.03000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    178	78    	[ -1.25474697e-02   1.09190000e+02]	[  4.05261672e-04   1.95803473e+00]	[ -1.47389933e-02   9.90000000e+01]	[ -1.23358617e-02   1.17000000e+02]
    179	80    	[ -1.25716762e-02   1.09390000e+02]	[  4.48158244e-04   2.07795573e+00]	[ -1.46615917e-02   1.00000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    180	69    	[ -1.24809502e-02   1.09160000e+02]	[  3.17734483e-04   1.22245654e+00]	[ -1.40897718e-02   1.04000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    181	71    	[ -1.25492737e-02   1.09140000e+02]	[  4.28486582e-04   1.61876496e+00]	[ -1.43988839e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    182	81    	[ -1.24917227e-02   1.08940000e+02]	[  3.51506197e-04   1.71941851e+00]	[ -1.37992001e-02   1.03000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    183	78    	[ -1.25043074e-02   1.09030000e+02]	[  3.25181733e-04   1.43146778e+00]	[ -1.35062467e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    184	83    	[ -1.24621864e-02   1.09070000e+02]	[  3.55762112e-04   1.51825558e+00]	[ -1.50221102e-02   1.05000000e+02]	[ -1.23358617e-02   1.19000000e+02]
    185	84    	[ -1.24741780e-02   1.08980000e+02]	[  3.81055133e-04   1.36367151e+00]	[ -1.48450294e-02   1.04000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    186	79    	[ -1.26124444e-02   1.08870000e+02]	[  6.30791356e-04   1.48091188e+00]	[ -1.62628352e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    187	77    	[ -1.24829770e-02   1.08890000e+02]	[  3.33338696e-04   1.32585821e+00]	[ -1.37540972e-02   1.02000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    188	81    	[ -1.24569030e-02   1.08950000e+02]	[  3.19093913e-04   1.54515371e+00]	[ -1.41784418e-02   1.03000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    189	67    	[ -1.24032763e-02   1.08920000e+02]	[  2.14321976e-04   9.01997783e-01]	[ -1.34320983e-02   1.02000000e+02]	[ -1.23358617e-02   1.11000000e+02]
    190	80    	[ -1.25747858e-02   1.08960000e+02]	[  4.56481091e-04   1.90745904e+00]	[ -1.49112690e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    191	71    	[ -1.25168054e-02   1.08780000e+02]	[  4.00606235e-04   1.71802212e+00]	[ -1.45304192e-02   1.00000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    192	69    	[ -1.24792702e-02   1.09290000e+02]	[  2.78941432e-04   1.29069749e+00]	[ -1.33151604e-02   1.06000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    193	72    	[ -1.25214932e-02   1.08840000e+02]	[  4.68576280e-04   1.24675579e+00]	[ -1.49732730e-02   1.03000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    194	82    	[ -1.24807592e-02   1.08760000e+02]	[  4.26408332e-04   1.54996774e+00]	[ -1.52911475e-02   1.03000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    195	69    	[ -1.25475276e-02   1.09240000e+02]	[  5.69333767e-04   2.07422275e+00]	[ -1.66560336e-02   1.02000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    196	66    	[ -1.24807850e-02   1.09050000e+02]	[  3.56998796e-04   9.63068014e-01]	[ -1.44455325e-02   1.03000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    197	64    	[ -1.25055201e-02   1.08900000e+02]	[  3.59332182e-04   1.36747943e+00]	[ -1.39862160e-02   1.01000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    198	62    	[ -1.25049207e-02   1.09170000e+02]	[  4.70716892e-04   8.60871651e-01]	[ -1.50391611e-02   1.06000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    199	72    	[ -1.25235691e-02   1.08910000e+02]	[  4.67348259e-04   1.54334053e+00]	[ -1.51770265e-02   1.00000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    200	74    	[ -1.25714046e-02   1.08910000e+02]	[  5.04111816e-04   1.58804912e+00]	[ -1.47786070e-02   1.04000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    201	69    	[ -1.25549262e-02   1.09050000e+02]	[  4.80217809e-04   1.51244835e+00]	[ -1.44858857e-02   1.04000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    202	64    	[ -1.25152411e-02   1.08950000e+02]	[  4.00232309e-04   1.27573508e+00]	[ -1.51345241e-02   1.02000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    203	75    	[ -1.26024862e-02   1.08950000e+02]	[  5.80756449e-04   1.51904575e+00]	[ -1.48707096e-02   1.04000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    204	76    	[ -1.25407562e-02   1.08950000e+02]	[  4.82660866e-04   1.60234204e+00]	[ -1.49098983e-02   1.03000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    205	80    	[ -1.24738932e-02   1.09120000e+02]	[  3.07315816e-04   1.39484766e+00]	[ -1.41875112e-02   1.03000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    206	75    	[ -1.24580864e-02   1.09080000e+02]	[  3.35546857e-04   1.13736538e+00]	[ -1.44759544e-02   1.02000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    207	89    	[ -1.24979701e-02   1.09230000e+02]	[  3.53127151e-04   9.47153631e-01]	[ -1.44013609e-02   1.07000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    208	71    	[ -1.24206527e-02   1.09190000e+02]	[  2.21091467e-04   1.12867179e+00]	[ -1.32802466e-02   1.05000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    209	76    	[ -1.25339467e-02   1.09130000e+02]	[  4.78428868e-04   1.55983974e+00]	[ -1.57553581e-02   1.02000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    210	82    	[ -1.25658572e-02   1.09060000e+02]	[  5.40488809e-04   1.77662602e+00]	[ -1.53855945e-02   1.03000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    211	81    	[ -1.24742428e-02   1.09100000e+02]	[  3.50457559e-04   1.17046999e+00]	[ -1.47244006e-02   1.04000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    212	81    	[ -1.25414318e-02   1.09010000e+02]	[  4.73575123e-04   1.26881835e+00]	[ -1.48854933e-02   1.02000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    213	89    	[ -1.24874437e-02   1.08930000e+02]	[  3.53293444e-04   1.32857066e+00]	[ -1.44637178e-02   1.03000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    214	75    	[ -1.24896203e-02   1.09020000e+02]	[  4.65346140e-04   1.10435502e+00]	[ -1.61718986e-02   1.04000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    215	79    	[ -1.24730456e-02   1.09050000e+02]	[  2.95846245e-04   1.18638105e+00]	[ -1.40021455e-02   1.05000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    216	72    	[ -1.26103709e-02   1.08980000e+02]	[  4.82320483e-04   1.96458647e+00]	[ -1.49306466e-02   1.02000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    217	77    	[ -1.24237107e-02   1.08940000e+02]	[  2.28644017e-04   1.62984662e+00]	[ -1.33820179e-02   9.90000000e+01]	[ -1.23358617e-02   1.15000000e+02]
    218	71    	[ -1.25473270e-02   1.08970000e+02]	[  6.18580079e-04   1.42446481e+00]	[ -1.65813038e-02   1.01000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    219	78    	[ -1.25235410e-02   1.09170000e+02]	[  3.93278928e-04   1.72658623e+00]	[ -1.44269660e-02   1.03000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    220	65    	[ -1.24878541e-02   1.08970000e+02]	[  3.53993422e-04   1.26059510e+00]	[ -1.47062040e-02   1.05000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    221	81    	[ -1.25596428e-02   1.08930000e+02]	[  5.49029007e-04   1.65683433e+00]	[ -1.54659188e-02   1.01000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    222	83    	[ -1.25539004e-02   1.08790000e+02]	[  4.85666115e-04   1.28292634e+00]	[ -1.48293212e-02   1.04000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    223	72    	[ -1.24510408e-02   1.09010000e+02]	[  3.17625762e-04   7.93662397e-01]	[ -1.44358924e-02   1.06000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    224	78    	[ -1.25343600e-02   1.09110000e+02]	[  4.34560044e-04   1.58047461e+00]	[ -1.47138892e-02   1.03000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    225	85    	[ -1.25148259e-02   1.09060000e+02]	[  4.22976770e-04   1.52852870e+00]	[ -1.50968575e-02   1.03000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    226	69    	[ -1.24859890e-02   1.08890000e+02]	[  3.75325584e-04   9.88888265e-01]	[ -1.49083037e-02   1.04000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    227	83    	[ -1.25326218e-02   1.08960000e+02]	[  4.26445743e-04   1.44858552e+00]	[ -1.44604078e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    228	74    	[ -1.25446583e-02   1.09040000e+02]	[  4.45178349e-04   1.31087757e+00]	[ -1.47273994e-02   1.03000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    229	72    	[ -1.24617663e-02   1.08940000e+02]	[  3.03823280e-04   1.27921851e+00]	[ -1.38225268e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    230	72    	[ -1.24403291e-02   1.09360000e+02]	[  2.89159193e-04   1.60324671e+00]	[ -1.37204949e-02   1.07000000e+02]	[ -1.23358617e-02   1.19000000e+02]
    231	81    	[ -1.25374414e-02   1.08890000e+02]	[  5.04406350e-04   1.74295726e+00]	[ -1.54222432e-02   1.02000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    232	80    	[ -1.25320487e-02   1.08870000e+02]	[  4.43564879e-04   1.47414382e+00]	[ -1.45253096e-02   1.03000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    233	74    	[ -1.24820546e-02   1.08960000e+02]	[  3.70029674e-04   1.36323146e+00]	[ -1.47741099e-02   1.04000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    234	74    	[ -1.25391844e-02   1.09300000e+02]	[  3.64271422e-04   1.76918060e+00]	[ -1.37014294e-02   1.03000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    235	87    	[ -1.25076995e-02   1.09150000e+02]	[  3.48429469e-04   1.57082781e+00]	[ -1.41579679e-02   1.03000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    236	83    	[ -1.25017213e-02   1.08930000e+02]	[  3.88552222e-04   1.78468485e+00]	[ -1.47355057e-02   1.02000000e+02]	[ -1.23358617e-02   1.20000000e+02]
    237	75    	[ -1.24830013e-02   1.09190000e+02]	[  2.95174250e-04   1.94779362e+00]	[ -1.36196860e-02   1.02000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    238	70    	[ -1.24926735e-02   1.08860000e+02]	[  4.54030305e-04   1.01014850e+00]	[ -1.51401061e-02   1.05000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    239	78    	[ -1.24885005e-02   1.09140000e+02]	[  3.54349763e-04   1.00019998e+00]	[ -1.43950177e-02   1.04000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    240	85    	[ -1.25293585e-02   1.09410000e+02]	[  4.37359615e-04   1.79496518e+00]	[ -1.44894990e-02   1.05000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    241	78    	[ -1.24864875e-02   1.08760000e+02]	[  3.49741820e-04   1.46369396e+00]	[ -1.42003489e-02   1.02000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    242	61    	[ -1.25002115e-02   1.09120000e+02]	[  3.92098243e-04   1.30598622e+00]	[ -1.47321864e-02   1.05000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    243	80    	[ -1.24879233e-02   1.09250000e+02]	[  3.03517605e-04   1.55804365e+00]	[ -1.35874967e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    244	69    	[ -1.25761822e-02   1.08940000e+02]	[  5.38612137e-04   1.36981751e+00]	[ -1.48139136e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    245	68    	[ -1.24869743e-02   1.08920000e+02]	[  3.83224686e-04   9.01997783e-01]	[ -1.46768412e-02   1.05000000e+02]	[ -1.23358617e-02   1.11000000e+02]
    246	73    	[ -1.25044594e-02   1.08840000e+02]	[  4.69830336e-04   1.29398609e+00]	[ -1.50496973e-02   1.01000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    247	76    	[ -1.24986324e-02   1.08940000e+02]	[  3.14968156e-04   1.63597066e+00]	[ -1.38022978e-02   1.02000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    248	74    	[ -1.25289948e-02   1.08950000e+02]	[  4.09608413e-04   1.18638105e+00]	[ -1.42932418e-02   1.05000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    249	68    	[ -1.25079003e-02   1.08890000e+02]	[  3.70910645e-04   1.82699206e+00]	[ -1.45660439e-02   9.90000000e+01]	[ -1.23358617e-02   1.14000000e+02]
    250	78    	[ -1.25502588e-02   1.09110000e+02]	[  4.14050640e-04   1.78826732e+00]	[ -1.45797206e-02   1.02000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    251	77    	[ -1.25064943e-02   1.09160000e+02]	[  3.44780510e-04   1.42632395e+00]	[ -1.40909569e-02   1.04000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    252	88    	[ -1.25224630e-02   1.09120000e+02]	[  3.56955843e-04   1.96102014e+00]	[ -1.36089984e-02   1.04000000e+02]	[ -1.23358617e-02   1.19000000e+02]
    253	71    	[ -1.25290838e-02   1.09110000e+02]	[  4.28686070e-04   1.69643744e+00]	[ -1.48455542e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    254	72    	[ -1.24956141e-02   1.08990000e+02]	[  4.25074696e-04   1.75211301e+00]	[ -1.50313634e-02   1.03000000e+02]	[ -1.23358617e-02   1.19000000e+02]
    255	78    	[ -1.25516777e-02   1.08790000e+02]	[  4.99938508e-04   1.47169970e+00]	[ -1.49857575e-02   1.02000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    256	81    	[ -1.24806555e-02   1.08860000e+02]	[  3.59170136e-04   1.24112852e+00]	[ -1.46384043e-02   1.04000000e+02]	[ -1.23358617e-02   1.12000000e+02]
    257	64    	[ -1.24717614e-02   1.08960000e+02]	[  3.02268977e-04   1.00915806e+00]	[ -1.40120608e-02   1.02000000e+02]	[ -1.23358617e-02   1.11000000e+02]
    258	78    	[ -1.25882209e-02   1.09090000e+02]	[  5.02956318e-04   1.81711309e+00]	[ -1.50533292e-02   1.03000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    259	74    	[ -1.24900254e-02   1.08830000e+02]	[  3.62619115e-04   1.42165397e+00]	[ -1.43488682e-02   1.00000000e+02]	[ -1.23358617e-02   1.13000000e+02]
    260	70    	[ -1.24565811e-02   1.08900000e+02]	[  2.87939924e-04   1.37477271e+00]	[ -1.40416938e-02   1.02000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    261	78    	[ -1.25916881e-02   1.08960000e+02]	[  5.16425011e-04   1.84889156e+00]	[ -1.54628619e-02   1.03000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    262	75    	[ -1.24963337e-02   1.08800000e+02]	[  3.40941667e-04   1.50996689e+00]	[ -1.38860308e-02   1.02000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    263	66    	[ -1.25309717e-02   1.08820000e+02]	[  4.74618786e-04   1.87285878e+00]	[ -1.46884327e-02   9.90000000e+01]	[ -1.23358617e-02   1.14000000e+02]
    264	74    	[ -1.24923372e-02   1.08950000e+02]	[  3.73579465e-04   1.25996032e+00]	[ -1.48469669e-02   1.02000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    265	81    	[ -1.25550554e-02   1.09310000e+02]	[  3.69436232e-04   2.05277860e+00]	[ -1.45640710e-02   1.04000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    266	81    	[ -1.24537408e-02   1.09150000e+02]	[  3.52782804e-04   1.05237826e+00]	[ -1.47627471e-02   1.06000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    267	79    	[ -1.24824952e-02   1.08990000e+02]	[  3.83035304e-04   1.46625373e+00]	[ -1.44227076e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    268	72    	[ -1.25104513e-02   1.09080000e+02]	[  4.00849765e-04   1.66541286e+00]	[ -1.41212809e-02   1.04000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    269	77    	[ -1.24162468e-02   1.09020000e+02]	[  2.24256609e-04   1.31893897e+00]	[ -1.32917659e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    270	63    	[ -1.24551062e-02   1.09030000e+02]	[  4.57923864e-04   1.33007519e+00]	[ -1.60125020e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    271	76    	[ -1.25367315e-02   1.09010000e+02]	[  5.46234743e-04   1.55881365e+00]	[ -1.61174241e-02   1.03000000e+02]	[ -1.23358617e-02   1.18000000e+02]
    272	74    	[ -1.24901032e-02   1.08880000e+02]	[  3.73279234e-04   1.19398492e+00]	[ -1.41141509e-02   1.05000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    273	79    	[ -1.25731693e-02   1.09240000e+02]	[  4.96694533e-04   1.75567651e+00]	[ -1.55673706e-02   1.02000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    274	70    	[ -1.25349043e-02   1.08770000e+02]	[  5.60528896e-04   1.59282767e+00]	[ -1.54916581e-02   1.01000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    275	68    	[ -1.24389305e-02   1.09000000e+02]	[  3.11084974e-04   9.48683298e-01]	[ -1.43263050e-02   1.04000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    276	75    	[ -1.24483124e-02   1.08930000e+02]	[  3.22167489e-04   1.53788816e+00]	[ -1.47304947e-02   1.01000000e+02]	[ -1.23358617e-02   1.15000000e+02]
    277	78    	[ -1.24836178e-02   1.09090000e+02]	[  3.43106272e-04   1.67985118e+00]	[ -1.47670982e-02   1.03000000e+02]	[ -1.23358617e-02   1.17000000e+02]
    278	77    	[ -1.24517536e-02   1.09030000e+02]	[  2.55468426e-04   1.03397292e+00]	[ -1.31933256e-02   1.05000000e+02]	[ -1.23358617e-02   1.14000000e+02]
    279	82    	[ -1.25444142e-02   1.09030000e+02]	[  3.95261006e-04   1.49969997e+00]	[ -1.46107417e-02   1.04000000e+02]	[ -1.23358617e-02   1.16000000e+02]
    280	83    	[ -1.25290057e-02   1.09050000e+02]	[  3.87197853e-04   1.82961745e+00]	[ -1.46186911e-02   1.01000000e+02]	[ -1.23270087e-02   1.16000000e+02]
    281	71    	[ -1.25156868e-02   1.09070000e+02]	[  4.07491543e-04   1.71029237e+00]	[ -1.48789110e-02   1.01000000e+02]	[ -1.23270087e-02   1.18000000e+02]
    282	79    	[ -1.24748553e-02   1.08980000e+02]	[  2.49177810e-04   9.05317624e-01]	[ -1.32998716e-02   1.04000000e+02]	[ -1.23270087e-02   1.12000000e+02]
    283	66    	[ -1.26057197e-02   1.09000000e+02]	[  4.78918263e-04   1.95448203e+00]	[ -1.51490342e-02   1.03000000e+02]	[ -1.23270087e-02   1.19000000e+02]
    284	85    	[ -1.26198847e-02   1.08950000e+02]	[  5.68095406e-04   1.91507180e+00]	[ -1.50225401e-02   9.90000000e+01]	[ -1.23270087e-02   1.16000000e+02]
    285	84    	[ -1.25956698e-02   1.08940000e+02]	[  4.80297018e-04   1.61133485e+00]	[ -1.52989724e-02   1.04000000e+02]	[ -1.23270087e-02   1.18000000e+02]
    286	68    	[ -1.25265085e-02   1.09100000e+02]	[  2.82499589e-04   1.55884573e+00]	[ -1.34338804e-02   1.03000000e+02]	[ -1.23270087e-02   1.18000000e+02]
    287	80    	[ -1.25545739e-02   1.09340000e+02]	[  3.44295198e-04   1.83423008e+00]	[ -1.44902096e-02   1.04000000e+02]	[ -1.23270087e-02   1.20000000e+02]
    288	66    	[ -1.25661339e-02   1.08730000e+02]	[  4.05425660e-04   1.20710397e+00]	[ -1.49089639e-02   1.04000000e+02]	[ -1.23270087e-02   1.11000000e+02]
    289	76    	[ -1.26134968e-02   1.08950000e+02]	[  4.15266136e-04   1.28354977e+00]	[ -1.47529593e-02   1.04000000e+02]	[ -1.23270087e-02   1.14000000e+02]
    290	74    	[ -1.25488464e-02   1.08900000e+02]	[  3.34770097e-04   1.62172747e+00]	[ -1.38521714e-02   1.02000000e+02]	[ -1.23270087e-02   1.15000000e+02]
    291	74    	[ -1.25585206e-02   1.08920000e+02]	[  5.22290423e-04   1.87445992e+00]	[ -1.46848885e-02   1.02000000e+02]	[ -1.23270087e-02   1.17000000e+02]
    292	80    	[ -1.25350931e-02   1.09030000e+02]	[  4.49843116e-04   1.49969997e+00]	[ -1.50295624e-02   1.05000000e+02]	[ -1.22728910e-02   1.15000000e+02]
    293	80    	[ -1.25696980e-02   1.09070000e+02]	[  5.92263143e-04   1.52482786e+00]	[ -1.54892259e-02   1.03000000e+02]	[ -1.22728910e-02   1.15000000e+02]
    294	71    	[ -1.25228742e-02   1.08720000e+02]	[  3.89004223e-04   1.95489130e+00]	[ -1.42796577e-02   1.01000000e+02]	[ -1.23270087e-02   1.16000000e+02]
    295	67    	[ -1.25797018e-02   1.08980000e+02]	[  5.01214921e-04   1.81096659e+00]	[ -1.50144163e-02   1.01000000e+02]	[ -1.23270087e-02   1.16000000e+02]
    296	78    	[ -1.25267755e-02   1.09120000e+02]	[  4.69872273e-04   1.75088549e+00]	[ -1.55094946e-02   1.01000000e+02]	[ -1.23270087e-02   1.15000000e+02]
    297	78    	[ -1.25108660e-02   1.08950000e+02]	[  4.49533721e-04   1.33697420e+00]	[ -1.47254757e-02   1.02000000e+02]	[ -1.23270087e-02   1.13000000e+02]
    298	64    	[ -1.24612105e-02   1.08920000e+02]	[  2.91425472e-04   1.14612390e+00]	[ -1.34819709e-02   1.04000000e+02]	[ -1.23270087e-02   1.14000000e+02]
    299	69    	[ -1.24407882e-02   1.08850000e+02]	[  2.73089188e-04   1.49916644e+00]	[ -1.36414236e-02   1.03000000e+02]	[ -1.23270087e-02   1.14000000e+02]
    300	75    	[ -1.24725840e-02   1.09110000e+02]	[  3.46044561e-04   1.18232821e+00]	[ -1.48606663e-02   1.02000000e+02]	[ -1.23270087e-02   1.14000000e+02]
    Selecting features with genetic algorithm.
    gen	nevals	avg                                	std                                	min                                	max                                
    0  	100   	[ -2.03431385e-02   1.09050000e+02]	[  2.99959730e-03   6.43952638e+00]	[ -3.06030752e-02   9.10000000e+01]	[ -1.59192553e-02   1.21000000e+02]
    1  	75    	[ -1.84909166e-02   1.10130000e+02]	[  2.00345812e-03   5.93743210e+00]	[ -2.50821161e-02   9.30000000e+01]	[ -1.58360105e-02   1.21000000e+02]
    2  	76    	[ -1.74453222e-02   1.11000000e+02]	[  1.94017162e-03   6.47610994e+00]	[ -3.19722851e-02   9.30000000e+01]	[ -1.50711844e-02   1.24000000e+02]
    3  	75    	[ -1.65439206e-02   1.13230000e+02]	[  9.45538330e-04   6.10877238e+00]	[ -2.16052989e-02   9.60000000e+01]	[ -1.50711844e-02   1.29000000e+02]
    4  	79    	[ -1.60301479e-02   1.14820000e+02]	[  6.06795559e-04   6.42865460e+00]	[ -1.79076601e-02   9.60000000e+01]	[ -1.45882204e-02   1.29000000e+02]
    5  	69    	[ -1.55795111e-02   1.18130000e+02]	[  6.02850928e-04   6.09861460e+00]	[ -1.82201328e-02   1.05000000e+02]	[ -1.46034898e-02   1.31000000e+02]
    6  	77    	[ -1.53118865e-02   1.19080000e+02]	[  5.04087964e-04   5.88163243e+00]	[ -1.75910456e-02   1.06000000e+02]	[ -1.44454754e-02   1.32000000e+02]
    7  	83    	[ -1.50327805e-02   1.21100000e+02]	[  5.56921009e-04   5.84380013e+00]	[ -1.71679879e-02   1.03000000e+02]	[ -1.39991767e-02   1.34000000e+02]
    8  	72    	[ -1.47268360e-02   1.23310000e+02]	[  3.64604672e-04   5.76141476e+00]	[ -1.63772284e-02   1.09000000e+02]	[ -1.39991767e-02   1.42000000e+02]
    9  	66    	[ -1.45743605e-02   1.24140000e+02]	[  3.43776744e-04   6.66936279e+00]	[ -1.57324609e-02   9.90000000e+01]	[ -1.39588282e-02   1.42000000e+02]
    10 	77    	[ -1.44633464e-02   1.26910000e+02]	[  6.13750732e-04   7.44324526e+00]	[ -1.94871901e-02   1.06000000e+02]	[ -1.38836408e-02   1.43000000e+02]
    11 	81    	[ -1.43169628e-02   1.27780000e+02]	[  3.54686524e-04   7.17437105e+00]	[ -1.58590363e-02   1.11000000e+02]	[ -1.38221579e-02   1.44000000e+02]
    12 	80    	[ -1.41443798e-02   1.26910000e+02]	[  2.92841218e-04   6.79719795e+00]	[ -1.54514269e-02   1.08000000e+02]	[ -1.36868699e-02   1.41000000e+02]
    13 	68    	[ -1.40281332e-02   1.27490000e+02]	[  3.27708025e-04   6.12942901e+00]	[ -1.60863770e-02   1.11000000e+02]	[ -1.36284979e-02   1.40000000e+02]
    14 	84    	[ -1.38974077e-02   1.26610000e+02]	[  1.77102337e-04   6.22076362e+00]	[ -1.48057136e-02   1.09000000e+02]	[ -1.36284979e-02   1.39000000e+02]
    15 	77    	[ -1.38762674e-02   1.25740000e+02]	[  3.70887591e-04   5.80106887e+00]	[ -1.60139351e-02   1.11000000e+02]	[ -1.35560009e-02   1.38000000e+02]
    16 	83    	[ -1.37807081e-02   1.27070000e+02]	[  2.85657906e-04   5.55743646e+00]	[ -1.59899351e-02   1.10000000e+02]	[ -1.34768135e-02   1.38000000e+02]
    17 	74    	[ -1.37241999e-02   1.26250000e+02]	[  2.39660918e-04   5.16599458e+00]	[ -1.53880614e-02   1.10000000e+02]	[ -1.34532675e-02   1.39000000e+02]
    18 	77    	[ -1.37107312e-02   1.26600000e+02]	[  2.62998255e-04   5.14975728e+00]	[ -1.46173449e-02   1.10000000e+02]	[ -1.33748166e-02   1.42000000e+02]
    19 	74    	[ -1.36779039e-02   1.26260000e+02]	[  2.66860316e-04   5.22803979e+00]	[ -1.49150501e-02   1.15000000e+02]	[ -1.33850056e-02   1.42000000e+02]
    20 	74    	[ -1.36247368e-02   1.25800000e+02]	[  3.07314536e-04   5.76194412e+00]	[ -1.55165525e-02   1.14000000e+02]	[ -1.33610304e-02   1.43000000e+02]
    21 	70    	[ -1.35554387e-02   1.26540000e+02]	[  2.21737743e-04   6.14885355e+00]	[ -1.45297065e-02   1.14000000e+02]	[ -1.33170578e-02   1.43000000e+02]
    22 	67    	[ -1.35953111e-02   1.25960000e+02]	[  3.91647021e-04   5.97648726e+00]	[ -1.57739900e-02   1.15000000e+02]	[ -1.33044280e-02   1.43000000e+02]
    23 	79    	[ -1.35091646e-02   1.24610000e+02]	[  3.25947754e-04   5.23621046e+00]	[ -1.48926115e-02   1.15000000e+02]	[ -1.33050040e-02   1.39000000e+02]
    24 	82    	[ -1.34708309e-02   1.23390000e+02]	[  4.04974652e-04   5.01177613e+00]	[ -1.66363725e-02   1.12000000e+02]	[ -1.32740109e-02   1.39000000e+02]
    25 	80    	[ -1.33711634e-02   1.22600000e+02]	[  1.90360743e-04   3.88844442e+00]	[ -1.50547014e-02   1.09000000e+02]	[ -1.32422118e-02   1.33000000e+02]
    26 	75    	[ -1.34274255e-02   1.21840000e+02]	[  3.70359350e-04   3.74090898e+00]	[ -1.54647118e-02   1.13000000e+02]	[ -1.32361611e-02   1.31000000e+02]
    27 	78    	[ -1.33430093e-02   1.21000000e+02]	[  2.15009508e-04   4.03980198e+00]	[ -1.48856419e-02   1.11000000e+02]	[ -1.32006248e-02   1.30000000e+02]
    28 	82    	[ -1.33725420e-02   1.20050000e+02]	[  3.11392064e-04   4.25294016e+00]	[ -1.50644389e-02   1.11000000e+02]	[ -1.31984550e-02   1.31000000e+02]
    29 	75    	[ -1.34061321e-02   1.19460000e+02]	[  4.06032991e-04   5.35988806e+00]	[ -1.58744144e-02   1.07000000e+02]	[ -1.31984550e-02   1.29000000e+02]
    30 	70    	[ -1.33902020e-02   1.19330000e+02]	[  4.72574361e-04   4.97806187e+00]	[ -1.57837312e-02   1.07000000e+02]	[ -1.31681582e-02   1.32000000e+02]
    31 	76    	[ -1.33577003e-02   1.18890000e+02]	[  3.97155141e-04   4.42920986e+00]	[ -1.61900642e-02   1.09000000e+02]	[ -1.31548439e-02   1.32000000e+02]
    32 	79    	[ -1.32774494e-02   1.17820000e+02]	[  2.47787467e-04   3.78253883e+00]	[ -1.48478799e-02   1.09000000e+02]	[ -1.31612730e-02   1.27000000e+02]
    33 	80    	[ -1.33854317e-02   1.17540000e+02]	[  5.15397758e-04   3.73743227e+00]	[ -1.69191817e-02   1.08000000e+02]	[ -1.31428669e-02   1.27000000e+02]
    34 	72    	[ -1.33129695e-02   1.16810000e+02]	[  3.78980852e-04   3.73816800e+00]	[ -1.51623834e-02   1.08000000e+02]	[ -1.31520376e-02   1.27000000e+02]
    35 	70    	[ -1.33132513e-02   1.16500000e+02]	[  3.26900246e-04   3.48568501e+00]	[ -1.45184433e-02   1.09000000e+02]	[ -1.31441215e-02   1.27000000e+02]
    36 	80    	[ -1.32722112e-02   1.16950000e+02]	[  2.86905064e-04   3.30265045e+00]	[ -1.50440425e-02   1.05000000e+02]	[ -1.31336924e-02   1.24000000e+02]
    37 	84    	[ -1.32489614e-02   1.16510000e+02]	[  2.42629623e-04   3.17960689e+00]	[ -1.45994991e-02   1.07000000e+02]	[ -1.31334159e-02   1.28000000e+02]
    38 	71    	[ -1.32801249e-02   1.15770000e+02]	[  4.29151563e-04   3.28284937e+00]	[ -1.64190455e-02   1.02000000e+02]	[ -1.31323073e-02   1.22000000e+02]
    39 	83    	[ -1.32361353e-02   1.15910000e+02]	[  2.87007300e-04   3.15307786e+00]	[ -1.48590010e-02   1.08000000e+02]	[ -1.31253125e-02   1.22000000e+02]
    40 	79    	[ -1.33260680e-02   1.15580000e+02]	[  6.10727434e-04   3.29296219e+00]	[ -1.84789870e-02   1.09000000e+02]	[ -1.31207551e-02   1.23000000e+02]
    41 	74    	[ -1.32355129e-02   1.15750000e+02]	[  3.65676278e-04   3.40404172e+00]	[ -1.62394775e-02   1.05000000e+02]	[ -1.31207551e-02   1.25000000e+02]
    42 	68    	[ -1.32811369e-02   1.14530000e+02]	[  3.84132914e-04   3.09016181e+00]	[ -1.57416732e-02   1.04000000e+02]	[ -1.31166729e-02   1.21000000e+02]
    43 	69    	[ -1.32128043e-02   1.15200000e+02]	[  2.33718235e-04   2.98328678e+00]	[ -1.42788138e-02   1.07000000e+02]	[ -1.31167721e-02   1.21000000e+02]
    44 	73    	[ -1.32290090e-02   1.14700000e+02]	[  2.70112709e-04   3.16385840e+00]	[ -1.42771876e-02   1.08000000e+02]	[ -1.31161029e-02   1.24000000e+02]
    45 	79    	[ -1.32476603e-02   1.14380000e+02]	[  3.45347857e-04   3.00925240e+00]	[ -1.52004002e-02   1.09000000e+02]	[ -1.31161029e-02   1.21000000e+02]
    46 	70    	[ -1.32828169e-02   1.13640000e+02]	[  4.24209255e-04   3.38974925e+00]	[ -1.54167678e-02   1.06000000e+02]	[ -1.31159031e-02   1.23000000e+02]
    47 	76    	[ -1.32269910e-02   1.12070000e+02]	[  3.09397509e-04   3.32341692e+00]	[ -1.51932986e-02   1.02000000e+02]	[ -1.31148870e-02   1.24000000e+02]
    48 	77    	[ -1.32041506e-02   1.11020000e+02]	[  2.85359721e-04   2.84246372e+00]	[ -1.54487202e-02   1.04000000e+02]	[ -1.31124468e-02   1.18000000e+02]
    49 	79    	[ -1.32164181e-02   1.10090000e+02]	[  2.74397731e-04   2.27637870e+00]	[ -1.48752924e-02   1.02000000e+02]	[ -1.31132886e-02   1.15000000e+02]
    50 	70    	[ -1.31974503e-02   1.10070000e+02]	[  2.02365835e-04   2.47893122e+00]	[ -1.41046206e-02   1.00000000e+02]	[ -1.31132885e-02   1.18000000e+02]
    51 	77    	[ -1.33025248e-02   1.09580000e+02]	[  3.92504120e-04   2.56974707e+00]	[ -1.49583601e-02   1.02000000e+02]	[ -1.31116140e-02   1.18000000e+02]
    52 	76    	[ -1.32849719e-02   1.08670000e+02]	[  6.59523147e-04   2.10739175e+00]	[ -1.88484173e-02   1.04000000e+02]	[ -1.31107970e-02   1.16000000e+02]
    53 	70    	[ -1.32213760e-02   1.08430000e+02]	[  2.89072279e-04   2.82932854e+00]	[ -1.44579469e-02   1.02000000e+02]	[ -1.31077782e-02   1.18000000e+02]
    54 	77    	[ -1.31941180e-02   1.08170000e+02]	[  2.87816575e-04   2.33261656e+00]	[ -1.53822075e-02   1.03000000e+02]	[ -1.31077782e-02   1.15000000e+02]
    55 	85    	[ -1.32035826e-02   1.07830000e+02]	[  3.23923109e-04   2.38350582e+00]	[ -1.50857040e-02   1.00000000e+02]	[ -1.31039501e-02   1.15000000e+02]
    56 	75    	[ -1.31980388e-02   1.07350000e+02]	[  2.78946536e-04   2.56271341e+00]	[ -1.49004120e-02   9.90000000e+01]	[ -1.31034399e-02   1.15000000e+02]
    57 	71    	[ -1.31854311e-02   1.07550000e+02]	[  2.00433416e-04   2.43464576e+00]	[ -1.40558695e-02   1.01000000e+02]	[ -1.31031413e-02   1.13000000e+02]
    58 	81    	[ -1.32477928e-02   1.07200000e+02]	[  3.72603417e-04   2.45356883e+00]	[ -1.51201395e-02   1.01000000e+02]	[ -1.31031340e-02   1.14000000e+02]
    59 	81    	[ -1.31882617e-02   1.06840000e+02]	[  2.25816065e-04   1.92727787e+00]	[ -1.43147819e-02   1.03000000e+02]	[ -1.31031340e-02   1.12000000e+02]
    60 	77    	[ -1.32146670e-02   1.06960000e+02]	[  2.52112167e-04   2.01950489e+00]	[ -1.41988582e-02   1.02000000e+02]	[ -1.31031340e-02   1.12000000e+02]
    61 	72    	[ -1.31933661e-02   1.06900000e+02]	[  2.66046858e-04   2.01246118e+00]	[ -1.48639532e-02   1.01000000e+02]	[ -1.30940204e-02   1.13000000e+02]
    62 	72    	[ -1.31878960e-02   1.06550000e+02]	[  2.69259109e-04   2.19715725e+00]	[ -1.47912441e-02   1.00000000e+02]	[ -1.30940204e-02   1.18000000e+02]
    63 	72    	[ -1.31938853e-02   1.07160000e+02]	[  2.73443405e-04   1.90641024e+00]	[ -1.48926968e-02   1.03000000e+02]	[ -1.30940252e-02   1.17000000e+02]
    64 	81    	[ -1.31795771e-02   1.06980000e+02]	[  2.69116408e-04   1.58101233e+00]	[ -1.49919587e-02   1.02000000e+02]	[ -1.30940234e-02   1.13000000e+02]
    65 	72    	[ -1.32149029e-02   1.06630000e+02]	[  2.97465882e-04   1.64106673e+00]	[ -1.46938138e-02   1.03000000e+02]	[ -1.30940234e-02   1.12000000e+02]
    66 	64    	[ -1.31715436e-02   1.06500000e+02]	[  2.10076276e-04   1.57797338e+00]	[ -1.46733748e-02   1.02000000e+02]	[ -1.30940046e-02   1.11000000e+02]
    67 	82    	[ -1.32464808e-02   1.06610000e+02]	[  4.12002446e-04   1.84875634e+00]	[ -1.59494229e-02   1.01000000e+02]	[ -1.30757575e-02   1.11000000e+02]
    68 	72    	[ -1.31785628e-02   1.06570000e+02]	[  3.15995579e-04   1.89343603e+00]	[ -1.51419048e-02   1.02000000e+02]	[ -1.30757575e-02   1.15000000e+02]
    69 	76    	[ -1.32034541e-02   1.06470000e+02]	[  3.42748103e-04   2.01223756e+00]	[ -1.51849377e-02   1.02000000e+02]	[ -1.30666693e-02   1.15000000e+02]
    70 	66    	[ -1.31633299e-02   1.06610000e+02]	[  2.91964870e-04   1.65466009e+00]	[ -1.55102320e-02   1.03000000e+02]	[ -1.30666684e-02   1.11000000e+02]
    71 	74    	[ -1.32704539e-02   1.05910000e+02]	[  4.04722106e-04   2.68735930e+00]	[ -1.50685795e-02   9.70000000e+01]	[ -1.30666682e-02   1.15000000e+02]
    72 	81    	[ -1.31417888e-02   1.06060000e+02]	[  2.20676436e-04   2.03872509e+00]	[ -1.48853893e-02   1.02000000e+02]	[ -1.30666682e-02   1.12000000e+02]
    73 	68    	[ -1.31556186e-02   1.05450000e+02]	[  2.45263501e-04   1.93584607e+00]	[ -1.45450536e-02   9.70000000e+01]	[ -1.30666681e-02   1.11000000e+02]
    74 	72    	[ -1.31560855e-02   1.05070000e+02]	[  2.58175261e-04   2.03103422e+00]	[ -1.46504872e-02   9.60000000e+01]	[ -1.30666681e-02   1.13000000e+02]
    75 	75    	[ -1.31396387e-02   1.04190000e+02]	[  2.38929588e-04   1.80385698e+00]	[ -1.47410010e-02   9.50000000e+01]	[ -1.30666667e-02   1.09000000e+02]
    76 	77    	[ -1.31899589e-02   1.04220000e+02]	[  2.97327247e-04   2.15675682e+00]	[ -1.45109404e-02   9.60000000e+01]	[ -1.30666667e-02   1.11000000e+02]
    77 	68    	[ -1.31988445e-02   1.03840000e+02]	[  3.77475388e-04   1.69540556e+00]	[ -1.48573000e-02   9.80000000e+01]	[ -1.30666667e-02   1.09000000e+02]
    78 	84    	[ -1.32031670e-02   1.04200000e+02]	[  3.57476674e-04   1.98997487e+00]	[ -1.50682168e-02   9.60000000e+01]	[ -1.30666667e-02   1.10000000e+02]
    79 	85    	[ -1.31891837e-02   1.04080000e+02]	[  2.71626898e-04   1.89039678e+00]	[ -1.46005989e-02   1.00000000e+02]	[ -1.30666667e-02   1.10000000e+02]
    80 	75    	[ -1.32436651e-02   1.03330000e+02]	[  4.14663955e-04   1.78356385e+00]	[ -1.56792080e-02   9.80000000e+01]	[ -1.30666667e-02   1.09000000e+02]
    81 	78    	[ -1.31846801e-02   1.03160000e+02]	[  2.92225942e-04   2.00858159e+00]	[ -1.47349209e-02   9.40000000e+01]	[ -1.30666667e-02   1.09000000e+02]
    82 	76    	[ -1.31697526e-02   1.03210000e+02]	[  2.94249025e-04   1.94059269e+00]	[ -1.46260127e-02   9.60000000e+01]	[ -1.30666667e-02   1.11000000e+02]
    83 	79    	[ -1.31631712e-02   1.03180000e+02]	[  3.20983884e-04   1.56448074e+00]	[ -1.50636970e-02   9.60000000e+01]	[ -1.30666667e-02   1.08000000e+02]
    84 	69    	[ -1.31696759e-02   1.03130000e+02]	[  3.05454920e-04   1.65320900e+00]	[ -1.45580977e-02   9.70000000e+01]	[ -1.30666667e-02   1.12000000e+02]
    85 	84    	[ -1.32427562e-02   1.03250000e+02]	[  4.18845948e-04   1.89934199e+00]	[ -1.51000709e-02   9.50000000e+01]	[ -1.30666667e-02   1.09000000e+02]
    86 	66    	[ -1.31436634e-02   1.03030000e+02]	[  2.51582422e-04   1.33757243e+00]	[ -1.46361168e-02   9.70000000e+01]	[ -1.30666667e-02   1.12000000e+02]
    87 	77    	[ -1.31748753e-02   1.03130000e+02]	[  3.61818221e-04   1.44675499e+00]	[ -1.54205060e-02   9.90000000e+01]	[ -1.30666667e-02   1.10000000e+02]
    88 	74    	[ -1.31570911e-02   1.03150000e+02]	[  2.41677267e-04   1.33697420e+00]	[ -1.45916467e-02   9.90000000e+01]	[ -1.30666667e-02   1.10000000e+02]
    89 	82    	[ -1.31220646e-02   1.03300000e+02]	[  1.68874427e-04   1.30766968e+00]	[ -1.42858857e-02   1.00000000e+02]	[ -1.30666667e-02   1.09000000e+02]
    90 	80    	[ -1.32343562e-02   1.03190000e+02]	[  5.75994915e-04   1.42614866e+00]	[ -1.72983802e-02   9.80000000e+01]	[ -1.30634850e-02   1.09000000e+02]
    91 	81    	[ -1.31509265e-02   1.02960000e+02]	[  3.10077950e-04   1.26427845e+00]	[ -1.50896342e-02   9.70000000e+01]	[ -1.30634850e-02   1.09000000e+02]
    92 	77    	[ -1.31565584e-02   1.03140000e+02]	[  2.45744754e-04   8.94650770e-01]	[ -1.43807561e-02   9.90000000e+01]	[ -1.30635013e-02   1.08000000e+02]
    93 	70    	[ -1.31893964e-02   1.03190000e+02]	[  3.34288151e-04   1.16357209e+00]	[ -1.47124132e-02   1.01000000e+02]	[ -1.30635012e-02   1.10000000e+02]
    94 	64    	[ -1.31562548e-02   1.03390000e+02]	[  2.62573652e-04   1.27196698e+00]	[ -1.46330960e-02   9.90000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    95 	74    	[ -1.31612100e-02   1.03550000e+02]	[  2.91494878e-04   1.52561463e+00]	[ -1.48597106e-02   9.70000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    96 	78    	[ -1.31289139e-02   1.03770000e+02]	[  2.34585769e-04   1.72542748e+00]	[ -1.45846127e-02   9.80000000e+01]	[ -1.30635012e-02   1.12000000e+02]
    97 	85    	[ -1.31328287e-02   1.03900000e+02]	[  2.31250444e-04   1.62788206e+00]	[ -1.44169113e-02   1.01000000e+02]	[ -1.30635012e-02   1.13000000e+02]
    98 	81    	[ -1.31802231e-02   1.03390000e+02]	[  3.28024514e-04   1.49596123e+00]	[ -1.49529692e-02   9.70000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    99 	70    	[ -1.31674563e-02   1.03270000e+02]	[  2.68589774e-04   1.76553108e+00]	[ -1.50288743e-02   9.70000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    100	73    	[ -1.31707482e-02   1.03380000e+02]	[  3.37355260e-04   1.60486760e+00]	[ -1.48136801e-02   9.70000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    101	69    	[ -1.31334199e-02   1.02880000e+02]	[  1.94348120e-04   1.13384302e+00]	[ -1.40844004e-02   9.50000000e+01]	[ -1.30635012e-02   1.07000000e+02]
    102	72    	[ -1.32127723e-02   1.02890000e+02]	[  4.68811189e-04   1.15667627e+00]	[ -1.64635169e-02   9.80000000e+01]	[ -1.30635012e-02   1.07000000e+02]
    103	71    	[ -1.31300489e-02   1.03110000e+02]	[  2.35148515e-04   1.31829435e+00]	[ -1.48191762e-02   9.80000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    104	72    	[ -1.32332942e-02   1.03000000e+02]	[  4.31066535e-04   1.80000000e+00]	[ -1.55251083e-02   9.40000000e+01]	[ -1.30635012e-02   1.11000000e+02]
    105	81    	[ -1.32114370e-02   1.03150000e+02]	[  3.90749397e-04   1.28354977e+00]	[ -1.51614794e-02   1.00000000e+02]	[ -1.30635012e-02   1.11000000e+02]
    106	74    	[ -1.32012322e-02   1.03170000e+02]	[  3.90313626e-04   1.53658713e+00]	[ -1.54756225e-02   9.60000000e+01]	[ -1.30635012e-02   1.08000000e+02]
    107	72    	[ -1.32260747e-02   1.03270000e+02]	[  4.11921460e-04   1.90186750e+00]	[ -1.56169462e-02   9.60000000e+01]	[ -1.30635012e-02   1.11000000e+02]
    108	81    	[ -1.32497551e-02   1.03050000e+02]	[  5.71695279e-04   1.50582203e+00]	[ -1.76773007e-02   9.80000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    109	75    	[ -1.31901128e-02   1.03110000e+02]	[  3.55262245e-04   1.47577098e+00]	[ -1.53685759e-02   9.70000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    110	79    	[ -1.31490426e-02   1.03170000e+02]	[  2.65492468e-04   1.47006803e+00]	[ -1.48549158e-02   9.70000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    111	86    	[ -1.31626059e-02   1.03040000e+02]	[  2.69343843e-04   1.33356665e+00]	[ -1.44450065e-02   9.90000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    112	74    	[ -1.31184324e-02   1.03290000e+02]	[  1.96989578e-04   9.82802116e-01]	[ -1.44729649e-02   1.03000000e+02]	[ -1.30635012e-02   1.09000000e+02]
    113	76    	[ -1.31524166e-02   1.03050000e+02]	[  2.70303211e-04   1.51904575e+00]	[ -1.47215834e-02   9.80000000e+01]	[ -1.30635012e-02   1.08000000e+02]
    114	80    	[ -1.31645146e-02   1.03060000e+02]	[  3.26504751e-04   1.33281657e+00]	[ -1.55561407e-02   9.70000000e+01]	[ -1.30635012e-02   1.07000000e+02]
    115	75    	[ -1.31621753e-02   1.03180000e+02]	[  3.12317536e-04   1.59612030e+00]	[ -1.48167473e-02   9.80000000e+01]	[ -1.30635012e-02   1.12000000e+02]
    116	75    	[ -1.31281268e-02   1.03110000e+02]	[  2.06190569e-04   1.15667627e+00]	[ -1.44547909e-02   1.00000000e+02]	[ -1.30635012e-02   1.08000000e+02]
    117	75    	[ -1.31680138e-02   1.03060000e+02]	[  2.89311709e-04   1.61752898e+00]	[ -1.45796988e-02   9.80000000e+01]	[ -1.30635012e-02   1.12000000e+02]
    118	77    	[ -1.31638380e-02   1.03020000e+02]	[  2.82899494e-04   1.19983332e+00]	[ -1.47033197e-02   9.70000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    119	78    	[ -1.31908426e-02   1.03140000e+02]	[  3.44092828e-04   1.28856509e+00]	[ -1.55060550e-02   9.90000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    120	70    	[ -1.31394248e-02   1.02920000e+02]	[  2.35957911e-04   7.44043010e-01]	[ -1.47339681e-02   9.80000000e+01]	[ -1.30635012e-02   1.06000000e+02]
    121	74    	[ -1.31526385e-02   1.03150000e+02]	[  4.66721235e-04   1.14345966e+00]	[ -1.75208112e-02   1.00000000e+02]	[ -1.30635012e-02   1.09000000e+02]
    122	86    	[ -1.31557915e-02   1.02990000e+02]	[  3.13061704e-04   1.39638820e+00]	[ -1.52207576e-02   9.60000000e+01]	[ -1.30635012e-02   1.07000000e+02]
    123	80    	[ -1.31525057e-02   1.02920000e+02]	[  2.23866495e-04   1.39053946e+00]	[ -1.44680563e-02   9.80000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    124	80    	[ -1.31987373e-02   1.02920000e+02]	[  3.57630875e-04   1.14612390e+00]	[ -1.55412586e-02   9.90000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    125	65    	[ -1.31566483e-02   1.03020000e+02]	[  3.23719183e-04   1.25682139e+00]	[ -1.54811192e-02   9.70000000e+01]	[ -1.30635012e-02   1.11000000e+02]
    126	74    	[ -1.31882895e-02   1.03150000e+02]	[  3.09166001e-04   1.24398553e+00]	[ -1.44996347e-02   9.70000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    127	85    	[ -1.31885129e-02   1.02950000e+02]	[  3.42490795e-04   1.13468057e+00]	[ -1.46340798e-02   9.80000000e+01]	[ -1.30635012e-02   1.07000000e+02]
    128	76    	[ -1.32104362e-02   1.03110000e+02]	[  3.46103108e-04   1.46215594e+00]	[ -1.50917739e-02   9.80000000e+01]	[ -1.30635012e-02   1.08000000e+02]
    129	71    	[ -1.31755876e-02   1.03310000e+02]	[  5.15578098e-04   1.21404283e+00]	[ -1.76968949e-02   1.00000000e+02]	[ -1.30635012e-02   1.10000000e+02]
    130	74    	[ -1.31714528e-02   1.03160000e+02]	[  3.47811963e-04   1.41223228e+00]	[ -1.53125432e-02   9.80000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    131	76    	[ -1.31516435e-02   1.03230000e+02]	[  2.25821938e-04   1.89132229e+00]	[ -1.42452173e-02   9.60000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    132	84    	[ -1.32510631e-02   1.02980000e+02]	[  4.34586514e-04   1.62468458e+00]	[ -1.57694965e-02   9.70000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    133	72    	[ -1.32217019e-02   1.02960000e+02]	[  3.74325264e-04   1.65481117e+00]	[ -1.51687214e-02   9.40000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    134	79    	[ -1.31937150e-02   1.03160000e+02]	[  3.98245563e-04   1.39799857e+00]	[ -1.58640939e-02   9.60000000e+01]	[ -1.30635012e-02   1.07000000e+02]
    135	70    	[ -1.31804883e-02   1.02900000e+02]	[  3.29921451e-04   1.46628783e+00]	[ -1.52401279e-02   9.70000000e+01]	[ -1.30635012e-02   1.11000000e+02]
    136	79    	[ -1.31831666e-02   1.03260000e+02]	[  3.61699722e-04   1.59135163e+00]	[ -1.55636057e-02   9.70000000e+01]	[ -1.30635012e-02   1.12000000e+02]
    137	74    	[ -1.31702052e-02   1.02920000e+02]	[  2.92926441e-04   9.55824252e-01]	[ -1.49657659e-02   9.80000000e+01]	[ -1.30635012e-02   1.07000000e+02]
    138	77    	[ -1.31869588e-02   1.02950000e+02]	[  3.46019396e-04   1.45859521e+00]	[ -1.47860131e-02   9.60000000e+01]	[ -1.30635012e-02   1.12000000e+02]
    139	76    	[ -1.31843525e-02   1.02970000e+02]	[  3.78773346e-04   1.53918810e+00]	[ -1.52379304e-02   9.60000000e+01]	[ -1.30635012e-02   1.10000000e+02]
    140	67    	[ -1.31608585e-02   1.03070000e+02]	[  2.56938150e-04   1.40893577e+00]	[ -1.43812502e-02   9.70000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    141	86    	[ -1.31953178e-02   1.03110000e+02]	[  3.67491214e-04   1.48926156e+00]	[ -1.50084920e-02   9.60000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    142	71    	[ -1.32155337e-02   1.03350000e+02]	[  3.99229682e-04   1.96659604e+00]	[ -1.53919281e-02   9.70000000e+01]	[ -1.30635012e-02   1.13000000e+02]
    143	75    	[ -1.31989283e-02   1.03070000e+02]	[  4.00048140e-04   1.64471882e+00]	[ -1.55749608e-02   9.80000000e+01]	[ -1.30635012e-02   1.11000000e+02]
    144	84    	[ -1.31689944e-02   1.03110000e+02]	[  2.45173260e-04   1.75439448e+00]	[ -1.40495929e-02   9.50000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    145	78    	[ -1.30997465e-02   1.02540000e+02]	[  1.60274801e-04   1.40299679e+00]	[ -1.44604052e-02   9.30000000e+01]	[ -1.30635012e-02   1.07000000e+02]
    146	82    	[ -1.31585226e-02   1.02330000e+02]	[  2.95251571e-04   1.31950748e+00]	[ -1.47145378e-02   9.80000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    147	72    	[ -1.32120922e-02   1.02260000e+02]	[  6.32193489e-04   1.64085344e+00]	[ -1.87203622e-02   9.50000000e+01]	[ -1.30635012e-02   1.09000000e+02]
    148	70    	[ -1.31184806e-02   1.02350000e+02]	[  1.99019589e-04   1.37386317e+00]	[ -1.44603389e-02   1.00000000e+02]	[ -1.30635012e-02   1.09000000e+02]
    149	81    	[ -1.31215495e-02   1.02190000e+02]	[  2.20598102e-04   9.86863719e-01]	[ -1.46478924e-02   1.00000000e+02]	[ -1.30635012e-02   1.08000000e+02]
    150	83    	[ -1.31661919e-02   1.02250000e+02]	[  2.44940708e-04   1.71099386e+00]	[ -1.45725464e-02   9.80000000e+01]	[ -1.30635012e-02   1.12000000e+02]
    151	75    	[ -1.31780178e-02   1.01960000e+02]	[  3.62739254e-04   1.14821601e+00]	[ -1.52023959e-02   9.80000000e+01]	[ -1.30635011e-02   1.06000000e+02]
    152	69    	[ -1.31320410e-02   1.02310000e+02]	[  2.18618358e-04   1.82589704e+00]	[ -1.45498024e-02   9.90000000e+01]	[ -1.30635011e-02   1.11000000e+02]
    153	83    	[ -1.31932384e-02   1.02500000e+02]	[  2.99987434e-04   1.94164878e+00]	[ -1.46614850e-02   9.80000000e+01]	[ -1.30635011e-02   1.11000000e+02]
    154	82    	[ -1.31406525e-02   1.02320000e+02]	[  2.41340701e-04   1.67260276e+00]	[ -1.45427883e-02   9.80000000e+01]	[ -1.30626357e-02   1.10000000e+02]
    155	69    	[ -1.32434294e-02   1.02060000e+02]	[  4.49053659e-04   2.08240246e+00]	[ -1.59003120e-02   9.30000000e+01]	[ -1.30624931e-02   1.11000000e+02]
    156	82    	[ -1.31525361e-02   1.01860000e+02]	[  2.33896286e-04   1.50346267e+00]	[ -1.42521064e-02   9.50000000e+01]	[ -1.30624929e-02   1.09000000e+02]
    157	64    	[ -1.32015889e-02   1.01850000e+02]	[  3.91774334e-04   1.86212244e+00]	[ -1.52508556e-02   9.50000000e+01]	[ -1.30624929e-02   1.11000000e+02]
    158	79    	[ -1.31903750e-02   1.01830000e+02]	[  2.93450592e-04   1.61279261e+00]	[ -1.42762768e-02   9.80000000e+01]	[ -1.30624929e-02   1.08000000e+02]
    159	66    	[ -1.31997453e-02   1.02260000e+02]	[  4.33734760e-04   2.07663189e+00]	[ -1.59090354e-02   9.70000000e+01]	[ -1.30624929e-02   1.10000000e+02]
    160	81    	[ -1.32120851e-02   1.02300000e+02]	[  3.72220455e-04   2.00748599e+00]	[ -1.52839662e-02   9.50000000e+01]	[ -1.30624929e-02   1.09000000e+02]
    161	64    	[ -1.31829781e-02   1.02800000e+02]	[  3.59600832e-04   1.54272486e+00]	[ -1.54650121e-02   9.70000000e+01]	[ -1.30624929e-02   1.09000000e+02]
    162	80    	[ -1.31884519e-02   1.02710000e+02]	[  3.90894607e-04   1.63275840e+00]	[ -1.61543197e-02   9.80000000e+01]	[ -1.30624929e-02   1.10000000e+02]
    163	84    	[ -1.31673313e-02   1.02390000e+02]	[  2.53370732e-04   2.03418288e+00]	[ -1.41131804e-02   9.50000000e+01]	[ -1.30624929e-02   1.14000000e+02]
    164	75    	[ -1.31870934e-02   1.02160000e+02]	[  3.85510805e-04   1.39082709e+00]	[ -1.58618124e-02   9.70000000e+01]	[ -1.30624929e-02   1.09000000e+02]
    165	79    	[ -1.31814195e-02   1.02110000e+02]	[  3.00865143e-04   1.67269244e+00]	[ -1.45391169e-02   9.50000000e+01]	[ -1.30624929e-02   1.08000000e+02]
    166	72    	[ -1.31590108e-02   1.02150000e+02]	[  3.01032236e-04   1.16081868e+00]	[ -1.46193493e-02   1.00000000e+02]	[ -1.30624929e-02   1.10000000e+02]
    167	77    	[ -1.31359638e-02   1.02060000e+02]	[  2.69098700e-04   1.04709121e+00]	[ -1.47985440e-02   1.00000000e+02]	[ -1.30624929e-02   1.07000000e+02]
    168	76    	[ -1.31366415e-02   1.01610000e+02]	[  2.29095509e-04   1.12156141e+00]	[ -1.46102600e-02   9.70000000e+01]	[ -1.30624929e-02   1.07000000e+02]
    169	72    	[ -1.31721771e-02   1.01250000e+02]	[  3.47794387e-04   1.54515371e+00]	[ -1.50941982e-02   9.20000000e+01]	[ -1.30624929e-02   1.06000000e+02]
    170	76    	[ -1.32264313e-02   1.01030000e+02]	[  6.63585625e-04   1.58401389e+00]	[ -1.90300494e-02   9.30000000e+01]	[ -1.30624929e-02   1.07000000e+02]
    171	83    	[ -1.31318818e-02   1.01200000e+02]	[  1.96335512e-04   1.28062485e+00]	[ -1.43868538e-02   9.70000000e+01]	[ -1.30624929e-02   1.07000000e+02]
    172	77    	[ -1.31620176e-02   1.01190000e+02]	[  2.94224776e-04   1.69525809e+00]	[ -1.45657075e-02   9.50000000e+01]	[ -1.30624929e-02   1.10000000e+02]
    173	88    	[ -1.31788674e-02   1.01050000e+02]	[  3.30577330e-04   1.17792190e+00]	[ -1.52284132e-02   9.70000000e+01]	[ -1.30624929e-02   1.06000000e+02]
    174	74    	[ -1.31676784e-02   1.01230000e+02]	[  2.97473587e-04   1.64228499e+00]	[ -1.48391953e-02   9.30000000e+01]	[ -1.30624929e-02   1.07000000e+02]
    175	73    	[ -1.31834265e-02   1.01430000e+02]	[  2.76910747e-04   1.91444509e+00]	[ -1.42919824e-02   9.50000000e+01]	[ -1.30624929e-02   1.10000000e+02]
    176	74    	[ -1.31557258e-02   1.01450000e+02]	[  3.19438426e-04   1.42390309e+00]	[ -1.58057360e-02   9.80000000e+01]	[ -1.30624928e-02   1.09000000e+02]
    177	71    	[ -1.31731670e-02   1.01290000e+02]	[  2.76676074e-04   1.43034961e+00]	[ -1.47431023e-02   9.70000000e+01]	[ -1.30624928e-02   1.11000000e+02]
    178	81    	[ -1.31828108e-02   1.01440000e+02]	[  3.43845971e-04   1.47864803e+00]	[ -1.54868675e-02   9.70000000e+01]	[ -1.30624928e-02   1.07000000e+02]
    179	79    	[ -1.31340769e-02   1.01580000e+02]	[  1.97991449e-04   1.88244522e+00]	[ -1.39446686e-02   9.60000000e+01]	[ -1.30624928e-02   1.10000000e+02]
    180	84    	[ -1.31349710e-02   1.01720000e+02]	[  2.02246945e-04   1.93432159e+00]	[ -1.40129889e-02   9.60000000e+01]	[ -1.30624928e-02   1.11000000e+02]
    181	80    	[ -1.31723417e-02   1.02080000e+02]	[  2.77027448e-04   1.58543369e+00]	[ -1.46138731e-02   9.60000000e+01]	[ -1.30624928e-02   1.08000000e+02]
    182	69    	[ -1.31407211e-02   1.02060000e+02]	[  2.90266906e-04   1.16464587e+00]	[ -1.50962750e-02   9.70000000e+01]	[ -1.30624928e-02   1.08000000e+02]
    183	76    	[ -1.31335400e-02   1.02320000e+02]	[  2.09107968e-04   1.47566934e+00]	[ -1.43756328e-02   9.80000000e+01]	[ -1.30624928e-02   1.09000000e+02]
    184	84    	[ -1.32090306e-02   1.02070000e+02]	[  5.44605740e-04   1.43704558e+00]	[ -1.76981994e-02   9.50000000e+01]	[ -1.30624928e-02   1.07000000e+02]
    185	79    	[ -1.31837795e-02   1.02070000e+02]	[  4.43419132e-04   5.33947563e-01]	[ -1.55317713e-02   9.90000000e+01]	[ -1.30624928e-02   1.05000000e+02]
    186	67    	[ -1.31347510e-02   1.02140000e+02]	[  2.64693893e-04   1.67343957e+00]	[ -1.47999673e-02   9.60000000e+01]	[ -1.30624928e-02   1.11000000e+02]
    187	75    	[ -1.31149441e-02   1.02180000e+02]	[  1.71782420e-04   1.20316250e+00]	[ -1.42951962e-02   9.80000000e+01]	[ -1.30624928e-02   1.08000000e+02]
    188	75    	[ -1.31826588e-02   1.02160000e+02]	[  3.68391482e-04   1.36176356e+00]	[ -1.51316596e-02   9.70000000e+01]	[ -1.30624928e-02   1.09000000e+02]
    189	76    	[ -1.31940869e-02   1.02200000e+02]	[  2.92905944e-04   1.74355958e+00]	[ -1.45801862e-02   9.70000000e+01]	[ -1.30624928e-02   1.11000000e+02]
    190	78    	[ -1.32022593e-02   1.01900000e+02]	[  4.19374899e-04   2.14242853e+00]	[ -1.57989377e-02   9.10000000e+01]	[ -1.30624928e-02   1.10000000e+02]
    191	87    	[ -1.31735446e-02   1.02010000e+02]	[  4.02920757e-04   1.60309076e+00]	[ -1.65496244e-02   9.40000000e+01]	[ -1.30624928e-02   1.09000000e+02]
    192	73    	[ -1.31495045e-02   1.02250000e+02]	[  2.65743395e-04   1.53215534e+00]	[ -1.46113044e-02   9.60000000e+01]	[ -1.30624928e-02   1.07000000e+02]
    193	76    	[ -1.32668379e-02   1.02150000e+02]	[  5.13868692e-04   1.80762275e+00]	[ -1.60927900e-02   9.60000000e+01]	[ -1.30624928e-02   1.13000000e+02]
    194	80    	[ -1.31605715e-02   1.02090000e+02]	[  2.63877848e-04   1.49060390e+00]	[ -1.45113993e-02   9.60000000e+01]	[ -1.30624928e-02   1.10000000e+02]
    195	73    	[ -1.31921635e-02   1.02100000e+02]	[  3.26791566e-04   1.27671453e+00]	[ -1.47006277e-02   9.80000000e+01]	[ -1.30624928e-02   1.07000000e+02]
    196	76    	[ -1.31787724e-02   1.02040000e+02]	[  3.88457944e-04   1.08554134e+00]	[ -1.53203336e-02   9.70000000e+01]	[ -1.30624928e-02   1.06000000e+02]
    197	70    	[ -1.31509583e-02   1.02160000e+02]	[  2.39499529e-04   2.01851431e+00]	[ -1.45765556e-02   9.10000000e+01]	[ -1.30624928e-02   1.10000000e+02]
    198	77    	[ -1.32041844e-02   1.02250000e+02]	[  6.57268875e-04   1.30671343e+00]	[ -1.93202120e-02   9.70000000e+01]	[ -1.30624928e-02   1.09000000e+02]
    199	69    	[ -1.31420256e-02   1.02060000e+02]	[  2.44139331e-04   9.36162379e-01]	[ -1.44024590e-02   9.90000000e+01]	[ -1.30624928e-02   1.08000000e+02]
    200	76    	[ -1.31822208e-02   1.02130000e+02]	[  3.25129576e-04   1.21371331e+00]	[ -1.47049898e-02   9.80000000e+01]	[ -1.30624928e-02   1.07000000e+02]
    201	83    	[ -1.31421387e-02   1.02160000e+02]	[  2.74753096e-04   1.17234807e+00]	[ -1.53906009e-02   1.00000000e+02]	[ -1.30624928e-02   1.10000000e+02]
    202	82    	[ -1.31516813e-02   1.02040000e+02]	[  2.14545457e-04   1.38506318e+00]	[ -1.41940024e-02   9.60000000e+01]	[ -1.30624928e-02   1.08000000e+02]
    203	73    	[ -1.31632726e-02   1.02170000e+02]	[  2.38798722e-04   1.12298709e+00]	[ -1.42398886e-02   9.70000000e+01]	[ -1.30624928e-02   1.07000000e+02]
    204	78    	[ -1.31346266e-02   1.01890000e+02]	[  1.81913395e-04   1.66069263e+00]	[ -1.41712309e-02   9.30000000e+01]	[ -1.30624928e-02   1.09000000e+02]
    205	79    	[ -1.31785582e-02   1.01970000e+02]	[  3.26384955e-04   1.18705518e+00]	[ -1.46324563e-02   9.50000000e+01]	[ -1.30624928e-02   1.06000000e+02]
    206	77    	[ -1.31796089e-02   1.02140000e+02]	[  2.83612123e-04   1.62493077e+00]	[ -1.46066296e-02   9.70000000e+01]	[ -1.30624928e-02   1.09000000e+02]
    207	77    	[ -1.31415972e-02   1.02200000e+02]	[  2.66224917e-04   1.45602198e+00]	[ -1.48783709e-02   9.70000000e+01]	[ -1.30624928e-02   1.10000000e+02]
    208	74    	[ -1.31604015e-02   1.02280000e+02]	[  2.54563404e-04   1.85515498e+00]	[ -1.43752517e-02   9.80000000e+01]	[ -1.30624928e-02   1.10000000e+02]
    209	68    	[ -1.31383127e-02   1.02100000e+02]	[  2.40056652e-04   1.30766968e+00]	[ -1.46859242e-02   9.60000000e+01]	[ -1.30624928e-02   1.11000000e+02]
    210	87    	[ -1.31705168e-02   1.02040000e+02]	[  2.81575203e-04   1.09471457e+00]	[ -1.42726346e-02   9.90000000e+01]	[ -1.30624928e-02   1.08000000e+02]
    211	73    	[ -1.31828607e-02   1.01950000e+02]	[  3.19829581e-04   1.40978722e+00]	[ -1.50689063e-02   9.40000000e+01]	[ -1.30624928e-02   1.06000000e+02]
    212	78    	[ -1.31539899e-02   1.02170000e+02]	[  2.83529908e-04   1.76099404e+00]	[ -1.48956332e-02   9.60000000e+01]	[ -1.30624928e-02   1.09000000e+02]
    213	75    	[ -1.31920212e-02   1.02390000e+02]	[  3.33036958e-04   1.52901929e+00]	[ -1.49936227e-02   9.80000000e+01]	[ -1.30624730e-02   1.11000000e+02]
    214	78    	[ -1.31758462e-02   1.02190000e+02]	[  3.42266029e-04   1.36157996e+00]	[ -1.58890548e-02   9.80000000e+01]	[ -1.30624730e-02   1.08000000e+02]
    215	62    	[ -1.31367398e-02   1.02270000e+02]	[  1.96835867e-04   1.73121345e+00]	[ -1.44518558e-02   9.70000000e+01]	[ -1.30624730e-02   1.10000000e+02]
    216	76    	[ -1.31097232e-02   1.02140000e+02]	[  1.45593642e-04   1.66144515e+00]	[ -1.39923744e-02   9.70000000e+01]	[ -1.30624730e-02   1.13000000e+02]
    217	76    	[ -1.31165154e-02   1.01960000e+02]	[  1.47741971e-04   1.42772546e+00]	[ -1.38371389e-02   9.80000000e+01]	[ -1.30577774e-02   1.08000000e+02]
    218	76    	[ -1.31616751e-02   1.01790000e+02]	[  3.17896391e-04   1.54463588e+00]	[ -1.50316806e-02   9.70000000e+01]	[ -1.30561653e-02   1.07000000e+02]
    219	80    	[ -1.31264303e-02   1.01320000e+02]	[  1.84530114e-04   1.61170717e+00]	[ -1.40050868e-02   9.40000000e+01]	[ -1.30560899e-02   1.08000000e+02]
    220	81    	[ -1.31889974e-02   1.01190000e+02]	[  3.60977300e-04   1.50794562e+00]	[ -1.47878744e-02   9.80000000e+01]	[ -1.30560866e-02   1.08000000e+02]
    221	66    	[ -1.31091856e-02   1.00870000e+02]	[  1.42526639e-04   1.43982638e+00]	[ -1.40218929e-02   9.60000000e+01]	[ -1.30560866e-02   1.06000000e+02]
    222	79    	[ -1.31655697e-02   1.00740000e+02]	[  2.78880202e-04   1.56601405e+00]	[ -1.47625736e-02   9.70000000e+01]	[ -1.30560865e-02   1.07000000e+02]
    223	80    	[ -1.31870812e-02   1.00890000e+02]	[  4.29521457e-04   1.90207781e+00]	[ -1.65464544e-02   9.40000000e+01]	[ -1.30560864e-02   1.08000000e+02]
    224	84    	[ -1.31345873e-02   1.00550000e+02]	[  2.11838393e-04   2.13717103e+00]	[ -1.44275666e-02   9.00000000e+01]	[ -1.30560864e-02   1.07000000e+02]
    225	78    	[ -1.31497689e-02   1.00200000e+02]	[  2.65771939e-04   1.43527001e+00]	[ -1.46225151e-02   9.50000000e+01]	[ -1.30560864e-02   1.06000000e+02]
    226	63    	[ -1.31838936e-02   9.99000000e+01]	[  3.69802686e-04   1.85202592e+00]	[ -1.51235368e-02   9.20000000e+01]	[ -1.30560864e-02   1.06000000e+02]
    227	81    	[ -1.31969238e-02   9.97300000e+01]	[  4.18735293e-04   1.87005348e+00]	[ -1.60393968e-02   9.10000000e+01]	[ -1.30560864e-02   1.06000000e+02]
    228	75    	[ -1.31915254e-02   1.00240000e+02]	[  3.75893354e-04   1.63780341e+00]	[ -1.54439718e-02   9.60000000e+01]	[ -1.30560864e-02   1.07000000e+02]
    229	84    	[ -1.32151033e-02   1.00240000e+02]	[  3.83149243e-04   1.87147001e+00]	[ -1.50254919e-02   9.30000000e+01]	[ -1.30560864e-02   1.08000000e+02]
    230	86    	[ -1.31953606e-02   1.00970000e+02]	[  4.00588270e-04   1.99226002e+00]	[ -1.55161050e-02   9.40000000e+01]	[ -1.30560864e-02   1.08000000e+02]
    231	88    	[ -1.31613148e-02   1.00460000e+02]	[  3.19916823e-04   1.50612085e+00]	[ -1.53362806e-02   9.60000000e+01]	[ -1.30560864e-02   1.10000000e+02]
    232	68    	[ -1.31829228e-02   1.00400000e+02]	[  3.46857662e-04   1.49666295e+00]	[ -1.48349069e-02   9.50000000e+01]	[ -1.30560864e-02   1.08000000e+02]
    233	83    	[ -1.31522812e-02   1.00540000e+02]	[  2.07821440e-04   2.10912304e+00]	[ -1.39715170e-02   9.60000000e+01]	[ -1.30560864e-02   1.09000000e+02]
    234	90    	[ -1.31501808e-02   1.00320000e+02]	[  3.04974201e-04   1.67857082e+00]	[ -1.55829596e-02   9.40000000e+01]	[ -1.30560864e-02   1.09000000e+02]
    235	65    	[ -1.31718125e-02   1.00100000e+02]	[  3.56764437e-04   7.14142843e-01]	[ -1.50549179e-02   9.70000000e+01]	[ -1.30560864e-02   1.04000000e+02]
    236	68    	[ -1.31554966e-02   1.00140000e+02]	[  2.57399196e-04   1.74367428e+00]	[ -1.44851952e-02   9.20000000e+01]	[ -1.30560864e-02   1.05000000e+02]
    237	67    	[ -1.31968993e-02   1.00420000e+02]	[  4.64255714e-04   1.99087920e+00]	[ -1.63863402e-02   9.30000000e+01]	[ -1.30560864e-02   1.09000000e+02]
    238	82    	[ -1.31406478e-02   1.00320000e+02]	[  2.45909057e-04   1.29522199e+00]	[ -1.46560789e-02   9.70000000e+01]	[ -1.30560864e-02   1.05000000e+02]
    239	72    	[ -1.32711713e-02   1.00290000e+02]	[  5.64680917e-04   1.94573893e+00]	[ -1.70402844e-02   9.50000000e+01]	[ -1.30560864e-02   1.11000000e+02]
    240	74    	[ -1.31586300e-02   1.00320000e+02]	[  3.04110071e-04   1.73712406e+00]	[ -1.52042345e-02   9.30000000e+01]	[ -1.30560864e-02   1.06000000e+02]
    241	87    	[ -1.31350135e-02   1.00170000e+02]	[  2.09103061e-04   1.31950748e+00]	[ -1.41600385e-02   9.50000000e+01]	[ -1.30560864e-02   1.06000000e+02]
    242	76    	[ -1.31647551e-02   1.00210000e+02]	[  2.46833093e-04   1.83463893e+00]	[ -1.43252708e-02   9.40000000e+01]	[ -1.30560864e-02   1.07000000e+02]
    243	72    	[ -1.31017356e-02   1.00030000e+02]	[  1.67875939e-04   1.35981616e+00]	[ -1.44863011e-02   9.50000000e+01]	[ -1.30560864e-02   1.06000000e+02]
    244	72    	[ -1.31089614e-02   1.00100000e+02]	[  1.80564326e-04   9.84885780e-01]	[ -1.40961875e-02   9.70000000e+01]	[ -1.30560864e-02   1.05000000e+02]
    245	82    	[ -1.32388708e-02   1.00080000e+02]	[  4.24198072e-04   1.61046577e+00]	[ -1.60326408e-02   9.20000000e+01]	[ -1.30560864e-02   1.06000000e+02]
    246	72    	[ -1.31934356e-02   1.00270000e+02]	[  4.06038297e-04   1.63006135e+00]	[ -1.62134883e-02   9.40000000e+01]	[ -1.30560864e-02   1.06000000e+02]
    247	81    	[ -1.31364158e-02   1.00270000e+02]	[  2.37212220e-04   1.63618459e+00]	[ -1.44430950e-02   9.60000000e+01]	[ -1.30560864e-02   1.09000000e+02]
    248	85    	[ -1.31211847e-02   9.99000000e+01]	[  2.24320135e-04   1.21243557e+00]	[ -1.42789173e-02   9.50000000e+01]	[ -1.30560864e-02   1.07000000e+02]
    249	75    	[ -1.31324949e-02   1.00090000e+02]	[  2.46634921e-04   1.28914700e+00]	[ -1.43692057e-02   9.50000000e+01]	[ -1.30555192e-02   1.06000000e+02]
    250	69    	[ -1.31509278e-02   9.99900000e+01]	[  2.92289425e-04   1.29996154e+00]	[ -1.48335032e-02   9.60000000e+01]	[ -1.30555188e-02   1.08000000e+02]
    251	79    	[ -1.31779215e-02   9.99800000e+01]	[  2.65013954e-04   1.54259522e+00]	[ -1.42830977e-02   9.50000000e+01]	[ -1.30555188e-02   1.04000000e+02]
    252	80    	[ -1.31385828e-02   9.98100000e+01]	[  2.45884070e-04   1.74180940e+00]	[ -1.48646909e-02   9.20000000e+01]	[ -1.30555188e-02   1.06000000e+02]
    253	68    	[ -1.31512538e-02   9.93300000e+01]	[  2.90321285e-04   1.88708770e+00]	[ -1.52130196e-02   9.30000000e+01]	[ -1.30555188e-02   1.07000000e+02]
    254	79    	[ -1.31896810e-02   9.93600000e+01]	[  2.93130547e-04   2.57106982e+00]	[ -1.43438290e-02   9.40000000e+01]	[ -1.30555188e-02   1.09000000e+02]
    255	77    	[ -1.31707255e-02   9.82600000e+01]	[  3.31386944e-04   1.57238672e+00]	[ -1.48478347e-02   9.10000000e+01]	[ -1.30555188e-02   1.04000000e+02]
    256	83    	[ -1.30983457e-02   9.78000000e+01]	[  1.80370973e-04   1.03923048e+00]	[ -1.44444831e-02   9.50000000e+01]	[ -1.30555188e-02   1.04000000e+02]
    257	83    	[ -1.31411412e-02   9.77900000e+01]	[  2.57802159e-04   1.73951143e+00]	[ -1.46800589e-02   9.50000000e+01]	[ -1.30555188e-02   1.04000000e+02]
    258	69    	[ -1.31569517e-02   9.73800000e+01]	[  3.21928649e-04   1.67797497e+00]	[ -1.56130538e-02   9.10000000e+01]	[ -1.30555188e-02   1.06000000e+02]
    259	76    	[ -1.31300370e-02   9.73300000e+01]	[  2.32945491e-04   1.58148664e+00]	[ -1.45599175e-02   9.10000000e+01]	[ -1.30555188e-02   1.06000000e+02]
    260	63    	[ -1.31564869e-02   9.75300000e+01]	[  2.78410214e-04   1.76892623e+00]	[ -1.44072455e-02   9.30000000e+01]	[ -1.30555188e-02   1.05000000e+02]
    261	66    	[ -1.31911202e-02   9.77100000e+01]	[  3.08635157e-04   2.09425404e+00]	[ -1.44774558e-02   9.30000000e+01]	[ -1.30555188e-02   1.06000000e+02]
    262	69    	[ -1.32013465e-02   9.73600000e+01]	[  3.89811937e-04   1.45958898e+00]	[ -1.49111355e-02   9.40000000e+01]	[ -1.30555188e-02   1.02000000e+02]
    263	77    	[ -1.31537205e-02   9.76000000e+01]	[  2.29807986e-04   2.03960781e+00]	[ -1.44686013e-02   9.40000000e+01]	[ -1.30555188e-02   1.08000000e+02]
    264	79    	[ -1.31534371e-02   9.73200000e+01]	[  3.10530221e-04   1.32574507e+00]	[ -1.53574173e-02   9.40000000e+01]	[ -1.30555188e-02   1.03000000e+02]
    265	82    	[ -1.31249743e-02   9.69700000e+01]	[  2.24841892e-04   1.01444566e+00]	[ -1.46403425e-02   9.30000000e+01]	[ -1.30555188e-02   1.01000000e+02]
    266	82    	[ -1.31722191e-02   9.70000000e+01]	[  2.93041446e-04   1.29614814e+00]	[ -1.48562131e-02   9.20000000e+01]	[ -1.30555188e-02   1.02000000e+02]
    267	90    	[ -1.32072150e-02   9.73100000e+01]	[  3.48956479e-04   1.35421564e+00]	[ -1.48877628e-02   9.30000000e+01]	[ -1.30555188e-02   1.02000000e+02]
    268	76    	[ -1.31496804e-02   9.75500000e+01]	[  2.57996496e-04   1.70513929e+00]	[ -1.46901900e-02   9.20000000e+01]	[ -1.30555188e-02   1.05000000e+02]
    269	77    	[ -1.32369958e-02   9.71400000e+01]	[  5.29326573e-04   1.41435498e+00]	[ -1.71749541e-02   9.10000000e+01]	[ -1.30555188e-02   1.03000000e+02]
    270	81    	[ -1.31820031e-02   9.72000000e+01]	[  2.89773525e-04   1.82756669e+00]	[ -1.44379754e-02   9.30000000e+01]	[ -1.30555188e-02   1.05000000e+02]
    271	78    	[ -1.31456409e-02   9.71200000e+01]	[  3.35129034e-04   1.47159777e+00]	[ -1.59781369e-02   9.30000000e+01]	[ -1.30555188e-02   1.06000000e+02]
    272	70    	[ -1.31990784e-02   9.71700000e+01]	[  4.11463561e-04   1.08678425e+00]	[ -1.53611140e-02   9.40000000e+01]	[ -1.30537278e-02   1.01000000e+02]
    273	86    	[ -1.31611459e-02   9.71300000e+01]	[  4.42945456e-04   1.48764915e+00]	[ -1.61295990e-02   9.40000000e+01]	[ -1.30533792e-02   1.06000000e+02]
    274	77    	[ -1.31502603e-02   9.72800000e+01]	[  2.82260714e-04   1.94463364e+00]	[ -1.52502728e-02   9.30000000e+01]	[ -1.30533792e-02   1.05000000e+02]
    275	92    	[ -1.32135938e-02   9.69200000e+01]	[  4.54261176e-04   1.77019773e+00]	[ -1.53525159e-02   8.90000000e+01]	[ -1.30533792e-02   1.06000000e+02]
    276	83    	[ -1.31539303e-02   9.66700000e+01]	[  2.66363754e-04   2.13098569e+00]	[ -1.45391630e-02   8.60000000e+01]	[ -1.30533792e-02   1.08000000e+02]
    277	68    	[ -1.31365034e-02   9.65800000e+01]	[  2.57536810e-04   1.63205392e+00]	[ -1.47096934e-02   9.20000000e+01]	[ -1.30533792e-02   1.04000000e+02]
    278	65    	[ -1.31168022e-02   9.61200000e+01]	[  2.30306522e-04   1.16000000e+00]	[ -1.45832617e-02   9.10000000e+01]	[ -1.30533792e-02   1.01000000e+02]
    279	78    	[ -1.31525017e-02   9.63300000e+01]	[  3.74951274e-04   1.69147864e+00]	[ -1.59391356e-02   9.00000000e+01]	[ -1.30533792e-02   1.05000000e+02]
    280	76    	[ -1.31556053e-02   9.64700000e+01]	[  3.30744751e-04   1.70560840e+00]	[ -1.51706347e-02   9.20000000e+01]	[ -1.30533792e-02   1.05000000e+02]
    281	80    	[ -1.31911186e-02   9.64100000e+01]	[  4.06808402e-04   1.72101714e+00]	[ -1.59346287e-02   9.30000000e+01]	[ -1.30533792e-02   1.05000000e+02]
    282	85    	[ -1.31589834e-02   9.65500000e+01]	[  3.61164880e-04   1.85135086e+00]	[ -1.61503616e-02   9.00000000e+01]	[ -1.30533792e-02   1.03000000e+02]
    283	80    	[ -1.31372363e-02   9.61900000e+01]	[  2.51540196e-04   1.07419737e+00]	[ -1.44736894e-02   9.30000000e+01]	[ -1.30533792e-02   1.02000000e+02]
    284	60    	[ -1.31350186e-02   9.64500000e+01]	[  2.48169640e-04   1.55804365e+00]	[ -1.43826250e-02   9.30000000e+01]	[ -1.30533792e-02   1.03000000e+02]
    285	82    	[ -1.31410712e-02   9.64600000e+01]	[  2.99385309e-04   1.93607851e+00]	[ -1.49334506e-02   9.30000000e+01]	[ -1.30533792e-02   1.05000000e+02]
    286	68    	[ -1.31641304e-02   9.63500000e+01]	[  2.85350503e-04   1.40267601e+00]	[ -1.49548519e-02   9.20000000e+01]	[ -1.30533792e-02   1.02000000e+02]
    287	80    	[ -1.32044763e-02   9.62100000e+01]	[  3.80387425e-04   1.38054337e+00]	[ -1.50418999e-02   9.20000000e+01]	[ -1.30533792e-02   1.01000000e+02]
    288	70    	[ -1.31970648e-02   9.63600000e+01]	[  3.53260702e-04   1.85752524e+00]	[ -1.48907022e-02   9.00000000e+01]	[ -1.30520042e-02   1.03000000e+02]
    289	82    	[ -1.32444485e-02   9.62100000e+01]	[  4.38737367e-04   1.79050272e+00]	[ -1.48404776e-02   9.10000000e+01]	[ -1.30520042e-02   1.04000000e+02]
    290	70    	[ -1.31314373e-02   9.65300000e+01]	[  1.96575154e-04   1.65804101e+00]	[ -1.41323170e-02   9.10000000e+01]	[ -1.30519858e-02   1.02000000e+02]
    291	82    	[ -1.31512716e-02   9.67700000e+01]	[  2.30181567e-04   1.77118604e+00]	[ -1.44579822e-02   9.10000000e+01]	[ -1.30519858e-02   1.04000000e+02]
    292	76    	[ -1.31985960e-02   9.69600000e+01]	[  3.46073586e-04   2.07807603e+00]	[ -1.53810812e-02   9.20000000e+01]	[ -1.30519858e-02   1.04000000e+02]
    293	79    	[ -1.31468800e-02   9.70300000e+01]	[  3.26420673e-04   2.06618005e+00]	[ -1.54036707e-02   9.20000000e+01]	[ -1.30518441e-02   1.06000000e+02]
    294	82    	[ -1.31713177e-02   9.70500000e+01]	[  3.41642055e-04   1.94615005e+00]	[ -1.49724908e-02   9.20000000e+01]	[ -1.30518441e-02   1.04000000e+02]
    295	77    	[ -1.31641490e-02   9.70800000e+01]	[  3.65735860e-04   1.69516961e+00]	[ -1.54336622e-02   9.30000000e+01]	[ -1.30518441e-02   1.04000000e+02]
    296	75    	[ -1.31774045e-02   9.68500000e+01]	[  4.05912649e-04   1.89934199e+00]	[ -1.57879170e-02   9.10000000e+01]	[ -1.30518441e-02   1.04000000e+02]
    297	80    	[ -1.32254650e-02   9.68400000e+01]	[  4.23017180e-04   1.59197990e+00]	[ -1.58531464e-02   9.10000000e+01]	[ -1.30518441e-02   1.04000000e+02]
    298	71    	[ -1.31726646e-02   9.65000000e+01]	[  3.42483436e-04   1.35277493e+00]	[ -1.49402711e-02   9.30000000e+01]	[ -1.30518441e-02   1.04000000e+02]
    299	74    	[ -1.31598281e-02   9.61000000e+01]	[  3.72969749e-04   1.46628783e+00]	[ -1.53993772e-02   9.20000000e+01]	[ -1.30518441e-02   1.03000000e+02]
    300	71    	[ -1.31590929e-02   9.57900000e+01]	[  2.94250600e-04   1.47169970e+00]	[ -1.44973443e-02   9.20000000e+01]	[ -1.30518441e-02   1.03000000e+02]
    Selecting features with genetic algorithm.
    gen	nevals	avg                                	std                                	min                                	max                                
    0  	100   	[ -1.85572089e-02   1.09840000e+02]	[  2.26471252e-03   8.27613436e+00]	[ -2.47868916e-02   8.90000000e+01]	[ -1.39457197e-02   1.28000000e+02]
    1  	79    	[ -1.70836359e-02   1.10860000e+02]	[  1.83877822e-03   7.60660240e+00]	[ -2.40764897e-02   9.10000000e+01]	[ -1.41254135e-02   1.28000000e+02]
    2  	72    	[ -1.58634685e-02   1.13200000e+02]	[  1.03881682e-03   6.63777071e+00]	[ -1.95307172e-02   9.70000000e+01]	[ -1.41254135e-02   1.26000000e+02]
    3  	73    	[ -1.51853182e-02   1.14450000e+02]	[  7.73006478e-04   6.06362103e+00]	[ -1.73987465e-02   1.00000000e+02]	[ -1.32910173e-02   1.29000000e+02]
    4  	79    	[ -1.47774499e-02   1.14120000e+02]	[  5.82888141e-04   5.95194086e+00]	[ -1.63527574e-02   9.60000000e+01]	[ -1.34280399e-02   1.27000000e+02]
    5  	89    	[ -1.46284987e-02   1.15120000e+02]	[  5.46583770e-04   5.96201308e+00]	[ -1.71198296e-02   9.80000000e+01]	[ -1.34566080e-02   1.32000000e+02]
    6  	78    	[ -1.43635574e-02   1.14920000e+02]	[  5.32601520e-04   5.67922530e+00]	[ -1.66992609e-02   9.80000000e+01]	[ -1.33825902e-02   1.26000000e+02]
    7  	77    	[ -1.42811242e-02   1.15160000e+02]	[  5.75278184e-04   4.96129015e+00]	[ -1.61378377e-02   9.90000000e+01]	[ -1.33854141e-02   1.27000000e+02]
    8  	82    	[ -1.41386685e-02   1.15100000e+02]	[  4.35863611e-04   5.87622328e+00]	[ -1.55081699e-02   9.90000000e+01]	[ -1.32890808e-02   1.31000000e+02]
    9  	80    	[ -1.40126564e-02   1.14810000e+02]	[  4.28549921e-04   5.99615710e+00]	[ -1.57104594e-02   1.02000000e+02]	[ -1.30229509e-02   1.33000000e+02]
    10 	74    	[ -1.39105616e-02   1.14310000e+02]	[  4.61482950e-04   6.15742641e+00]	[ -1.60674879e-02   1.01000000e+02]	[ -1.30229509e-02   1.33000000e+02]
    11 	84    	[ -1.37924621e-02   1.14160000e+02]	[  4.17915879e-04   5.79606763e+00]	[ -1.56643591e-02   1.00000000e+02]	[ -1.31942954e-02   1.26000000e+02]
    12 	74    	[ -1.36949353e-02   1.14250000e+02]	[  3.38359699e-04   5.70854622e+00]	[ -1.54592233e-02   9.90000000e+01]	[ -1.30630198e-02   1.27000000e+02]
    13 	76    	[ -1.36702657e-02   1.14160000e+02]	[  5.43873234e-04   5.18212312e+00]	[ -1.82160989e-02   1.04000000e+02]	[ -1.30505148e-02   1.27000000e+02]
    14 	69    	[ -1.35525753e-02   1.13300000e+02]	[  3.28405023e-04   5.45068803e+00]	[ -1.49517172e-02   1.04000000e+02]	[ -1.29737218e-02   1.27000000e+02]
    15 	80    	[ -1.35211724e-02   1.12270000e+02]	[  4.04313844e-04   6.11204548e+00]	[ -1.51959038e-02   9.60000000e+01]	[ -1.27605267e-02   1.38000000e+02]
    16 	75    	[ -1.35605107e-02   1.12480000e+02]	[  9.48816946e-04   5.76624661e+00]	[ -2.05109251e-02   9.80000000e+01]	[ -1.27428214e-02   1.24000000e+02]
    17 	67    	[ -1.32198222e-02   1.11990000e+02]	[  3.00444586e-04   5.71400910e+00]	[ -1.41313048e-02   9.60000000e+01]	[ -1.26765767e-02   1.24000000e+02]
    18 	74    	[ -1.32642908e-02   1.11710000e+02]	[  3.87187464e-04   7.05449502e+00]	[ -1.48841843e-02   9.60000000e+01]	[ -1.26765767e-02   1.28000000e+02]
    19 	55    	[ -1.31520862e-02   1.11130000e+02]	[  3.32038395e-04   6.87117894e+00]	[ -1.47523107e-02   9.60000000e+01]	[ -1.26765767e-02   1.27000000e+02]
    20 	66    	[ -1.31597131e-02   1.11880000e+02]	[  4.12656081e-04   5.63432338e+00]	[ -1.53083527e-02   9.80000000e+01]	[ -1.26765767e-02   1.23000000e+02]
    21 	56    	[ -1.32135515e-02   1.11490000e+02]	[  7.42596995e-04   5.11565245e+00]	[ -1.80254358e-02   9.60000000e+01]	[ -1.26765767e-02   1.24000000e+02]
    22 	77    	[ -1.31785891e-02   1.11720000e+02]	[  3.26372448e-04   4.45439109e+00]	[ -1.44803393e-02   1.01000000e+02]	[ -1.27234498e-02   1.19000000e+02]
    23 	83    	[ -1.32089394e-02   1.12480000e+02]	[  7.21266644e-04   4.08529069e+00]	[ -1.97391415e-02   9.90000000e+01]	[ -1.27605267e-02   1.23000000e+02]
    24 	64    	[ -1.30980762e-02   1.12310000e+02]	[  3.30610144e-04   3.83847626e+00]	[ -1.44884844e-02   1.01000000e+02]	[ -1.26497261e-02   1.24000000e+02]
    25 	86    	[ -1.31476699e-02   1.11880000e+02]	[  3.98166651e-04   4.16960430e+00]	[ -1.48457381e-02   9.90000000e+01]	[ -1.24671986e-02   1.22000000e+02]
    26 	69    	[ -1.31678568e-02   1.11240000e+02]	[  5.96926234e-04   3.47021613e+00]	[ -1.55716958e-02   1.01000000e+02]	[ -1.24671986e-02   1.19000000e+02]
    27 	84    	[ -1.31255255e-02   1.11200000e+02]	[  4.09910742e-04   2.71661554e+00]	[ -1.49722819e-02   1.03000000e+02]	[ -1.27059832e-02   1.19000000e+02]
    28 	81    	[ -1.30411118e-02   1.11740000e+02]	[  3.34011790e-04   2.46422402e+00]	[ -1.43312392e-02   1.06000000e+02]	[ -1.26226923e-02   1.19000000e+02]
    29 	79    	[ -1.30231152e-02   1.11970000e+02]	[  4.84981006e-04   2.49581650e+00]	[ -1.53666166e-02   1.06000000e+02]	[ -1.27557273e-02   1.24000000e+02]
    30 	83    	[ -1.30081890e-02   1.12170000e+02]	[  4.04683876e-04   1.71496356e+00]	[ -1.49132754e-02   1.04000000e+02]	[ -1.27557273e-02   1.18000000e+02]
    31 	85    	[ -1.30173770e-02   1.11940000e+02]	[  5.66163962e-04   1.58631649e+00]	[ -1.68375351e-02   1.07000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    32 	78    	[ -1.30023154e-02   1.11740000e+02]	[  4.57414744e-04   1.30858702e+00]	[ -1.47986189e-02   1.07000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    33 	79    	[ -1.29299058e-02   1.12040000e+02]	[  3.67452693e-04   1.37054734e+00]	[ -1.41949400e-02   1.07000000e+02]	[ -1.27605267e-02   1.21000000e+02]
    34 	75    	[ -1.29858742e-02   1.11940000e+02]	[  4.31118147e-04   1.80454981e+00]	[ -1.48759187e-02   1.04000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    35 	74    	[ -1.29983538e-02   1.11850000e+02]	[  5.14817791e-04   1.32193041e+00]	[ -1.53201320e-02   1.04000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    36 	71    	[ -1.29134558e-02   1.11900000e+02]	[  3.30807020e-04   1.23693169e+00]	[ -1.40914784e-02   1.05000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    37 	74    	[ -1.29364373e-02   1.12050000e+02]	[  4.60304481e-04   1.70513929e+00]	[ -1.50716798e-02   1.04000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    38 	80    	[ -1.29721485e-02   1.11900000e+02]	[  4.58220642e-04   1.51327460e+00]	[ -1.49347577e-02   1.04000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    39 	79    	[ -1.29718782e-02   1.12460000e+02]	[  3.62593243e-04   1.70540318e+00]	[ -1.42742822e-02   1.07000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    40 	70    	[ -1.29205918e-02   1.11940000e+02]	[  4.23307676e-04   1.47526269e+00]	[ -1.51658136e-02   1.06000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    41 	77    	[ -1.29085567e-02   1.12360000e+02]	[  4.42751814e-04   1.40370937e+00]	[ -1.51118026e-02   1.09000000e+02]	[ -1.27605267e-02   1.20000000e+02]
    42 	68    	[ -1.29456702e-02   1.12090000e+02]	[  4.14253946e-04   1.52377820e+00]	[ -1.51592551e-02   1.05000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    43 	85    	[ -1.29810251e-02   1.11850000e+02]	[  4.23781522e-04   1.53215534e+00]	[ -1.46540607e-02   1.05000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    44 	66    	[ -1.28889546e-02   1.11870000e+02]	[  3.21290309e-04   1.09229117e+00]	[ -1.41990640e-02   1.07000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    45 	68    	[ -1.29371813e-02   1.11950000e+02]	[  3.73881854e-04   8.17006732e-01]	[ -1.47493086e-02   1.09000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    46 	76    	[ -1.29247355e-02   1.12150000e+02]	[  4.76983392e-04   1.81865335e+00]	[ -1.60904757e-02   1.05000000e+02]	[ -1.27605267e-02   1.22000000e+02]
    47 	75    	[ -1.29165935e-02   1.11820000e+02]	[  4.28564424e-04   1.21144542e+00]	[ -1.50368626e-02   1.07000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    48 	82    	[ -1.29660641e-02   1.11870000e+02]	[  4.49534934e-04   1.54696477e+00]	[ -1.47283552e-02   1.02000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    49 	76    	[ -1.30246761e-02   1.12250000e+02]	[  5.18743651e-04   1.45172311e+00]	[ -1.49397559e-02   1.09000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    50 	76    	[ -1.29809513e-02   1.11770000e+02]	[  4.15780068e-04   1.83223907e+00]	[ -1.50175009e-02   1.03000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    51 	74    	[ -1.29293067e-02   1.12040000e+02]	[  4.10389928e-04   1.29553078e+00]	[ -1.49247489e-02   1.06000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    52 	73    	[ -1.29546032e-02   1.11680000e+02]	[  4.01032250e-04   1.52236658e+00]	[ -1.47335500e-02   1.05000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    53 	89    	[ -1.30772837e-02   1.11840000e+02]	[  5.83582246e-04   1.78168460e+00]	[ -1.53366751e-02   1.05000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    54 	74    	[ -1.29498717e-02   1.11900000e+02]	[  4.35336322e-04   1.17898261e+00]	[ -1.48894132e-02   1.06000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    55 	74    	[ -1.29639257e-02   1.11760000e+02]	[  3.95849537e-04   1.75567651e+00]	[ -1.44269947e-02   1.04000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    56 	77    	[ -1.28753921e-02   1.11900000e+02]	[  3.69116054e-04   8.77496439e-01]	[ -1.47943616e-02   1.06000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    57 	72    	[ -1.29326870e-02   1.12000000e+02]	[  3.52459088e-04   1.36381817e+00]	[ -1.45149474e-02   1.06000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    58 	83    	[ -1.30457264e-02   1.12020000e+02]	[  5.35720224e-04   1.96967002e+00]	[ -1.51939841e-02   1.06000000e+02]	[ -1.27605267e-02   1.22000000e+02]
    59 	75    	[ -1.29323600e-02   1.11910000e+02]	[  3.97190875e-04   9.70515327e-01]	[ -1.48280236e-02   1.06000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    60 	78    	[ -1.29525052e-02   1.11750000e+02]	[  3.65679731e-04   1.61477553e+00]	[ -1.42873781e-02   1.05000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    61 	70    	[ -1.29024767e-02   1.12140000e+02]	[  3.33722298e-04   1.44927568e+00]	[ -1.47718897e-02   1.07000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    62 	81    	[ -1.29226774e-02   1.12070000e+02]	[  3.85889058e-04   9.40797534e-01]	[ -1.51459417e-02   1.08000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    63 	59    	[ -1.28710357e-02   1.12040000e+02]	[  2.97036333e-04   1.13947356e+00]	[ -1.42957254e-02   1.06000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    64 	69    	[ -1.29371189e-02   1.11930000e+02]	[  4.29202119e-04   1.61403222e+00]	[ -1.51694498e-02   1.05000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    65 	75    	[ -1.29792091e-02   1.11890000e+02]	[  5.13799356e-04   1.01877377e+00]	[ -1.51325712e-02   1.06000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    66 	74    	[ -1.29422375e-02   1.11790000e+02]	[  4.86548307e-04   1.25932522e+00]	[ -1.61911986e-02   1.06000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    67 	72    	[ -1.29277136e-02   1.12100000e+02]	[  3.29288659e-04   1.23693169e+00]	[ -1.46731804e-02   1.08000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    68 	76    	[ -1.29601823e-02   1.11900000e+02]	[  4.43497218e-04   1.53948043e+00]	[ -1.51654198e-02   1.05000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    69 	87    	[ -1.30531995e-02   1.11750000e+02]	[  1.07486946e-03   1.45859521e+00]	[ -2.27431856e-02   1.06000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    70 	80    	[ -1.30024869e-02   1.12060000e+02]	[  4.67816012e-04   2.05338745e+00]	[ -1.49626597e-02   1.06000000e+02]	[ -1.27605267e-02   1.23000000e+02]
    71 	80    	[ -1.29390870e-02   1.11930000e+02]	[  3.87521024e-04   1.16837494e+00]	[ -1.44185115e-02   1.06000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    72 	72    	[ -1.30195328e-02   1.11890000e+02]	[  5.10317070e-04   1.48253162e+00]	[ -1.50198629e-02   1.07000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    73 	74    	[ -1.29666937e-02   1.11790000e+02]	[  4.02623011e-04   1.47847895e+00]	[ -1.46923124e-02   1.06000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    74 	64    	[ -1.29296744e-02   1.11800000e+02]	[  4.86841333e-04   1.24096736e+00]	[ -1.54961772e-02   1.04000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    75 	76    	[ -1.29900946e-02   1.11810000e+02]	[  4.64184212e-04   2.01342991e+00]	[ -1.48488349e-02   1.04000000e+02]	[ -1.27605267e-02   1.20000000e+02]
    76 	73    	[ -1.29734861e-02   1.12000000e+02]	[  4.18550575e-04   1.51657509e+00]	[ -1.46496389e-02   1.07000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    77 	83    	[ -1.29947996e-02   1.11790000e+02]	[  4.39803237e-04   2.48714696e+00]	[ -1.51826344e-02   9.90000000e+01]	[ -1.27605267e-02   1.20000000e+02]
    78 	75    	[ -1.29239148e-02   1.11880000e+02]	[  4.24411906e-04   1.48512626e+00]	[ -1.51086656e-02   1.04000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    79 	76    	[ -1.29166331e-02   1.11900000e+02]	[  3.18634796e-04   1.62788206e+00]	[ -1.41717467e-02   1.04000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    80 	75    	[ -1.29371641e-02   1.11860000e+02]	[  4.63307833e-04   1.21671689e+00]	[ -1.52695003e-02   1.06000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    81 	86    	[ -1.29844709e-02   1.12110000e+02]	[  4.76991761e-04   1.13925414e+00]	[ -1.58069209e-02   1.07000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    82 	75    	[ -1.29804415e-02   1.11740000e+02]	[  4.54743546e-04   1.37564530e+00]	[ -1.47411233e-02   1.06000000e+02]	[ -1.27572703e-02   1.16000000e+02]
    83 	80    	[ -1.29955799e-02   1.11620000e+02]	[  4.55875225e-04   1.91196234e+00]	[ -1.52334252e-02   1.02000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    84 	60    	[ -1.29159286e-02   1.11790000e+02]	[  3.68157566e-04   1.54463588e+00]	[ -1.43883369e-02   1.03000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    85 	79    	[ -1.30240849e-02   1.12000000e+02]	[  5.25347933e-04   1.21655251e+00]	[ -1.50397129e-02   1.09000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    86 	75    	[ -1.29390208e-02   1.11970000e+02]	[  3.80366237e-04   1.33757243e+00]	[ -1.48957104e-02   1.07000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    87 	84    	[ -1.29476798e-02   1.11840000e+02]	[  3.91251648e-04   1.44027775e+00]	[ -1.46092062e-02   1.06000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    88 	77    	[ -1.29756454e-02   1.11940000e+02]	[  4.87474903e-04   1.56729066e+00]	[ -1.49059727e-02   1.04000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    89 	81    	[ -1.29598876e-02   1.11800000e+02]	[  3.71501609e-04   1.29614814e+00]	[ -1.45244259e-02   1.05000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    90 	79    	[ -1.29716835e-02   1.12190000e+02]	[  4.23806058e-04   1.53424248e+00]	[ -1.46073105e-02   1.07000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    91 	89    	[ -1.29324331e-02   1.12050000e+02]	[  4.18762970e-04   1.86212244e+00]	[ -1.50434306e-02   1.02000000e+02]	[ -1.27605267e-02   1.20000000e+02]
    92 	71    	[ -1.29605040e-02   1.11820000e+02]	[  5.08870236e-04   1.11696016e+00]	[ -1.54463922e-02   1.07000000e+02]	[ -1.27605267e-02   1.15000000e+02]
    93 	80    	[ -1.29628179e-02   1.12020000e+02]	[  4.00775517e-04   1.53609896e+00]	[ -1.44193731e-02   1.07000000e+02]	[ -1.27605267e-02   1.20000000e+02]
    94 	63    	[ -1.28841114e-02   1.11660000e+02]	[  3.33550160e-04   1.40868733e+00]	[ -1.45492969e-02   1.01000000e+02]	[ -1.27605267e-02   1.13000000e+02]
    95 	78    	[ -1.29854069e-02   1.12160000e+02]	[  4.42138175e-04   1.48808602e+00]	[ -1.46831805e-02   1.08000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    96 	76    	[ -1.29077564e-02   1.12030000e+02]	[  3.48363088e-04   1.16150764e+00]	[ -1.45399831e-02   1.07000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    97 	77    	[ -1.29528882e-02   1.12090000e+02]	[  4.23577773e-04   1.31221187e+00]	[ -1.49158858e-02   1.06000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    98 	80    	[ -1.29622675e-02   1.11790000e+02]	[  4.09644827e-04   1.19411055e+00]	[ -1.50387399e-02   1.06000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    99 	72    	[ -1.29873195e-02   1.12050000e+02]	[  4.36490216e-04   1.23592071e+00]	[ -1.48276783e-02   1.08000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    100	72    	[ -1.29296766e-02   1.11740000e+02]	[  3.69136460e-04   1.44651305e+00]	[ -1.42992632e-02   1.05000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    101	75    	[ -1.29751967e-02   1.11810000e+02]	[  4.53155963e-04   1.92714815e+00]	[ -1.51061157e-02   1.02000000e+02]	[ -1.27605267e-02   1.18000000e+02]
    102	80    	[ -1.29313173e-02   1.12010000e+02]	[  3.76198753e-04   1.49996667e+00]	[ -1.47251938e-02   1.04000000e+02]	[ -1.27605267e-02   1.16000000e+02]
    103	82    	[ -1.29905354e-02   1.11790000e+02]	[  5.59460204e-04   1.66309952e+00]	[ -1.68194762e-02   1.04000000e+02]	[ -1.27605267e-02   1.17000000e+02]
    104	79    	[ -1.29103967e-02   1.11990000e+02]	[  3.28136854e-04   1.11798927e+00]	[ -1.45033129e-02   1.08000000e+02]	[ -1.27605267e-02   1.19000000e+02]
    105	75    	[ -1.29676609e-02   1.12050000e+02]	[  4.17624551e-04   1.71099386e+00]	[ -1.47835340e-02   1.05000000e+02]	[ -1.27605267e-02   1.20000000e+02]
    106	71    	[ -1.29007538e-02   1.12070000e+02]	[  3.64855860e-04   1.30579478e+00]	[ -1.46105670e-02   1.06000000e+02]	[ -1.27182396e-02   1.18000000e+02]
    107	71    	[ -1.29990068e-02   1.12200000e+02]	[  5.37658722e-04   1.52315462e+00]	[ -1.57846528e-02   1.08000000e+02]	[ -1.27182396e-02   1.21000000e+02]
    108	71    	[ -1.30330755e-02   1.12070000e+02]	[  4.73025427e-04   1.55083848e+00]	[ -1.46166559e-02   1.07000000e+02]	[ -1.27182396e-02   1.20000000e+02]
    109	66    	[ -1.30040926e-02   1.12000000e+02]	[  5.18568180e-04   1.07703296e+00]	[ -1.51970757e-02   1.08000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    110	76    	[ -1.29651268e-02   1.12010000e+02]	[  3.42793825e-04   1.37473634e+00]	[ -1.38902654e-02   1.08000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    111	71    	[ -1.29041387e-02   1.11690000e+02]	[  3.02004878e-04   1.38343775e+00]	[ -1.41957727e-02   1.06000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    112	74    	[ -1.29520913e-02   1.11950000e+02]	[  4.16455667e-04   1.26787223e+00]	[ -1.50847313e-02   1.08000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    113	77    	[ -1.29605516e-02   1.11950000e+02]	[  3.97102829e-04   1.53866826e+00]	[ -1.46673611e-02   1.06000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    114	70    	[ -1.29398896e-02   1.12040000e+02]	[  4.52738547e-04   1.42772546e+00]	[ -1.51269058e-02   1.07000000e+02]	[ -1.26938685e-02   1.21000000e+02]
    115	75    	[ -1.28762967e-02   1.11910000e+02]	[  3.23224091e-04   8.01186620e-01]	[ -1.44326239e-02   1.07000000e+02]	[ -1.26938685e-02   1.14000000e+02]
    116	79    	[ -1.30074433e-02   1.11810000e+02]	[  4.99633534e-04   1.42614866e+00]	[ -1.58858243e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    117	75    	[ -1.29103833e-02   1.12100000e+02]	[  3.52615083e-04   9.11043358e-01]	[ -1.45138355e-02   1.09000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    118	80    	[ -1.29456782e-02   1.12140000e+02]	[  2.94978195e-04   1.29630243e+00]	[ -1.40813077e-02   1.06000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    119	75    	[ -1.29950915e-02   1.12110000e+02]	[  3.81329567e-04   1.89681312e+00]	[ -1.49363970e-02   1.05000000e+02]	[ -1.26938685e-02   1.22000000e+02]
    120	80    	[ -1.30111204e-02   1.12230000e+02]	[  4.10143410e-04   1.19879106e+00]	[ -1.47386394e-02   1.08000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    121	71    	[ -1.29122456e-02   1.11950000e+02]	[  3.20552986e-04   1.20312094e+00]	[ -1.46703004e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    122	74    	[ -1.30893559e-02   1.11960000e+02]	[  5.17171514e-04   1.41364776e+00]	[ -1.59873542e-02   1.07000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    123	78    	[ -1.30956883e-02   1.11880000e+02]	[  4.93320352e-04   1.84000000e+00]	[ -1.59579763e-02   1.05000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    124	79    	[ -1.29739259e-02   1.11990000e+02]	[  3.13782235e-04   1.49996667e+00]	[ -1.38486444e-02   1.02000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    125	70    	[ -1.30374695e-02   1.11840000e+02]	[  5.04689899e-04   1.64754363e+00]	[ -1.51284457e-02   1.03000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    126	78    	[ -1.29820474e-02   1.11950000e+02]	[  4.05008200e-04   1.62095651e+00]	[ -1.47269361e-02   1.04000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    127	78    	[ -1.30137098e-02   1.11920000e+02]	[  4.15952149e-04   1.10163515e+00]	[ -1.44089714e-02   1.09000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    128	77    	[ -1.30185187e-02   1.11770000e+02]	[  5.15002441e-04   1.95885170e+00]	[ -1.54047996e-02   1.02000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    129	76    	[ -1.29141466e-02   1.12080000e+02]	[  3.97225304e-04   1.49452334e+00]	[ -1.42778105e-02   1.03000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    130	81    	[ -1.30092349e-02   1.12170000e+02]	[  6.30383980e-04   1.94964099e+00]	[ -1.59201271e-02   1.05000000e+02]	[ -1.26938685e-02   1.23000000e+02]
    131	78    	[ -1.28821625e-02   1.12280000e+02]	[  4.40289490e-04   1.38621788e+00]	[ -1.55473793e-02   1.08000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    132	82    	[ -1.28770564e-02   1.11890000e+02]	[  4.42677740e-04   1.29533779e+00]	[ -1.47640149e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    133	75    	[ -1.28596484e-02   1.11820000e+02]	[  4.25549335e-04   9.93780660e-01]	[ -1.49691025e-02   1.07000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    134	78    	[ -1.29003110e-02   1.11470000e+02]	[  4.33704437e-04   1.57133701e+00]	[ -1.46958196e-02   1.04000000e+02]	[ -1.26938685e-02   1.14000000e+02]
    135	84    	[ -1.29364080e-02   1.12020000e+02]	[  5.09979271e-04   1.37825977e+00]	[ -1.49053624e-02   1.07000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    136	75    	[ -1.28839018e-02   1.11940000e+02]	[  4.35394942e-04   1.45478521e+00]	[ -1.48506809e-02   1.07000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    137	73    	[ -1.29287578e-02   1.12000000e+02]	[  4.92006595e-04   1.57480157e+00]	[ -1.57524814e-02   1.07000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    138	79    	[ -1.28951779e-02   1.12090000e+02]	[  4.44813977e-04   1.27353838e+00]	[ -1.49014083e-02   1.06000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    139	73    	[ -1.29379819e-02   1.11800000e+02]	[  5.13842339e-04   1.32664992e+00]	[ -1.51579825e-02   1.06000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    140	73    	[ -1.28873144e-02   1.12110000e+02]	[  4.11101231e-04   1.59934362e+00]	[ -1.46043019e-02   1.05000000e+02]	[ -1.26938685e-02   1.21000000e+02]
    141	75    	[ -1.29946932e-02   1.11850000e+02]	[  5.88353665e-04   1.69926455e+00]	[ -1.56219136e-02   1.04000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    142	73    	[ -1.29761965e-02   1.12120000e+02]	[  5.19312243e-04   1.67499254e+00]	[ -1.50431427e-02   1.06000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    143	74    	[ -1.28737638e-02   1.12120000e+02]	[  3.88732687e-04   1.56384142e+00]	[ -1.41484109e-02   1.05000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    144	87    	[ -1.29414232e-02   1.12030000e+02]	[  4.63122680e-04   1.75188470e+00]	[ -1.47983887e-02   1.05000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    145	87    	[ -1.28998429e-02   1.11890000e+02]	[  4.28632266e-04   1.13044239e+00]	[ -1.44807780e-02   1.07000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    146	71    	[ -1.28480380e-02   1.12140000e+02]	[  3.48538432e-04   1.37854996e+00]	[ -1.41489691e-02   1.07000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    147	77    	[ -1.29868208e-02   1.12110000e+02]	[  5.25365141e-04   1.81049717e+00]	[ -1.52587159e-02   1.05000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    148	76    	[ -1.28738077e-02   1.12000000e+02]	[  4.14648844e-04   1.57480157e+00]	[ -1.45886368e-02   1.02000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    149	71    	[ -1.29394591e-02   1.12170000e+02]	[  5.02054853e-04   1.37153199e+00]	[ -1.51498334e-02   1.07000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    150	81    	[ -1.28921235e-02   1.11930000e+02]	[  5.69586325e-04   1.25900755e+00]	[ -1.71526586e-02   1.07000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    151	73    	[ -1.28980017e-02   1.11840000e+02]	[  4.78423822e-04   1.61071413e+00]	[ -1.52069001e-02   1.05000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    152	77    	[ -1.29455464e-02   1.11950000e+02]	[  4.49140759e-04   1.34443297e+00]	[ -1.48740414e-02   1.07000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    153	87    	[ -1.29304395e-02   1.12120000e+02]	[  4.30694374e-04   1.45794376e+00]	[ -1.40638728e-02   1.07000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    154	79    	[ -1.28473749e-02   1.11730000e+02]	[  3.47125289e-04   1.69620164e+00]	[ -1.42078903e-02   1.05000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    155	77    	[ -1.28944013e-02   1.11730000e+02]	[  5.59320214e-04   1.62391502e+00]	[ -1.65239173e-02   1.06000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    156	77    	[ -1.28357040e-02   1.12030000e+02]	[  3.33403406e-04   1.14415908e+00]	[ -1.40869637e-02   1.07000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    157	85    	[ -1.29479087e-02   1.11860000e+02]	[  4.98182117e-04   1.89219449e+00]	[ -1.53690491e-02   1.04000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    158	76    	[ -1.29453659e-02   1.12200000e+02]	[  4.97646393e-04   1.31148770e+00]	[ -1.52177270e-02   1.08000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    159	75    	[ -1.29328339e-02   1.11840000e+02]	[  4.94966730e-04   1.47458469e+00]	[ -1.50425206e-02   1.06000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    160	73    	[ -1.29031558e-02   1.11820000e+02]	[  4.82600495e-04   1.53218798e+00]	[ -1.49632731e-02   1.04000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    161	86    	[ -1.28758314e-02   1.11800000e+02]	[  3.92507970e-04   1.58745079e+00]	[ -1.45002518e-02   1.04000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    162	79    	[ -1.28939790e-02   1.11990000e+02]	[  4.91848355e-04   1.22061460e+00]	[ -1.53745446e-02   1.05000000e+02]	[ -1.26938685e-02   1.15000000e+02]
    163	76    	[ -1.28601367e-02   1.11830000e+02]	[  3.93077115e-04   1.24943987e+00]	[ -1.46537691e-02   1.06000000e+02]	[ -1.26938685e-02   1.15000000e+02]
    164	75    	[ -1.29635119e-02   1.11730000e+02]	[  5.05780718e-04   1.86469837e+00]	[ -1.47312981e-02   1.04000000e+02]	[ -1.26938685e-02   1.21000000e+02]
    165	74    	[ -1.28912485e-02   1.11930000e+02]	[  4.16134966e-04   1.35096262e+00]	[ -1.44792231e-02   1.06000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    166	80    	[ -1.28439171e-02   1.11930000e+02]	[  3.79038622e-04   1.05123737e+00]	[ -1.47073008e-02   1.08000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    167	77    	[ -1.28265940e-02   1.11920000e+02]	[  3.19802447e-04   9.66229786e-01]	[ -1.40795012e-02   1.06000000e+02]	[ -1.26938685e-02   1.15000000e+02]
    168	80    	[ -1.29146872e-02   1.12190000e+02]	[  4.63174623e-04   1.59809261e+00]	[ -1.47159390e-02   1.04000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    169	76    	[ -1.28940947e-02   1.11970000e+02]	[  4.45012557e-04   1.44537192e+00]	[ -1.48805364e-02   1.06000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    170	82    	[ -1.28285200e-02   1.12090000e+02]	[  3.28899448e-04   1.28914700e+00]	[ -1.38748934e-02   1.08000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    171	74    	[ -1.28943244e-02   1.12110000e+02]	[  5.17194172e-04   1.14799826e+00]	[ -1.52778375e-02   1.09000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    172	66    	[ -1.28540077e-02   1.11820000e+02]	[  3.93535401e-04   1.16944431e+00]	[ -1.46930156e-02   1.05000000e+02]	[ -1.26938685e-02   1.14000000e+02]
    173	75    	[ -1.29319193e-02   1.11810000e+02]	[  4.95508652e-04   1.59809261e+00]	[ -1.48633655e-02   1.03000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    174	62    	[ -1.28413470e-02   1.11940000e+02]	[  3.86593096e-04   1.42702488e+00]	[ -1.45096545e-02   1.06000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    175	70    	[ -1.29286782e-02   1.12170000e+02]	[  4.34300451e-04   1.51033109e+00]	[ -1.44166910e-02   1.08000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    176	77    	[ -1.29530265e-02   1.12050000e+02]	[  5.23448522e-04   1.51904575e+00]	[ -1.57139273e-02   1.05000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    177	81    	[ -1.29620535e-02   1.11860000e+02]	[  4.78119281e-04   1.51670696e+00]	[ -1.49488595e-02   1.05000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    178	79    	[ -1.29726049e-02   1.11440000e+02]	[  4.68927863e-04   1.78504902e+00]	[ -1.47524172e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    179	80    	[ -1.29240231e-02   1.11890000e+02]	[  4.76817803e-04   1.79941657e+00]	[ -1.52352946e-02   1.04000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    180	77    	[ -1.28744746e-02   1.11920000e+02]	[  4.20353323e-04   1.36879509e+00]	[ -1.51516218e-02   1.05000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    181	78    	[ -1.29225941e-02   1.11990000e+02]	[  4.51599602e-04   1.12689840e+00]	[ -1.45873137e-02   1.08000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    182	71    	[ -1.29351567e-02   1.11910000e+02]	[  5.59211445e-04   1.22552030e+00]	[ -1.58191952e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    183	75    	[ -1.27906448e-02   1.11780000e+02]	[  2.78863354e-04   1.20482364e+00]	[ -1.40539995e-02   1.04000000e+02]	[ -1.26938685e-02   1.14000000e+02]
    184	72    	[ -1.29427958e-02   1.11950000e+02]	[  5.61689817e-04   9.83615779e-01]	[ -1.61679934e-02   1.06000000e+02]	[ -1.26938685e-02   1.14000000e+02]
    185	71    	[ -1.28939999e-02   1.11970000e+02]	[  4.06144937e-04   1.17860087e+00]	[ -1.45074998e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    186	71    	[ -1.29136960e-02   1.12010000e+02]	[  4.79930932e-04   1.38920841e+00]	[ -1.48718750e-02   1.05000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    187	66    	[ -1.28821519e-02   1.11880000e+02]	[  3.84278106e-04   1.59549365e+00]	[ -1.44114695e-02   1.06000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    188	79    	[ -1.28661615e-02   1.11960000e+02]	[  4.07238206e-04   1.52918279e+00]	[ -1.46562658e-02   1.04000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    189	72    	[ -1.28821533e-02   1.12070000e+02]	[  4.04152318e-04   1.32857066e+00]	[ -1.46442864e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    190	75    	[ -1.28962913e-02   1.12000000e+02]	[  3.96537428e-04   1.53622915e+00]	[ -1.40470457e-02   1.06000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    191	68    	[ -1.29182806e-02   1.11970000e+02]	[  4.41710535e-04   1.47956075e+00]	[ -1.46045710e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    192	70    	[ -1.28667923e-02   1.12140000e+02]	[  4.16959683e-04   1.36396481e+00]	[ -1.47889055e-02   1.06000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    193	68    	[ -1.28492174e-02   1.11740000e+02]	[  4.05797253e-04   1.23790145e+00]	[ -1.50837003e-02   1.05000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    194	69    	[ -1.28567352e-02   1.12050000e+02]	[  4.65383316e-04   1.09886305e+00]	[ -1.58189226e-02   1.06000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    195	70    	[ -1.28748589e-02   1.12110000e+02]	[  4.15213162e-04   1.27196698e+00]	[ -1.50781148e-02   1.06000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    196	78    	[ -1.29456033e-02   1.12360000e+02]	[  5.65609252e-04   1.54609185e+00]	[ -1.59355163e-02   1.08000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    197	81    	[ -1.29836816e-02   1.11900000e+02]	[  4.33004353e-04   1.52643375e+00]	[ -1.42730296e-02   1.05000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    198	74    	[ -1.29832801e-02   1.11930000e+02]	[  5.96268462e-04   1.25103957e+00]	[ -1.67535975e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    199	82    	[ -1.30055778e-02   1.11980000e+02]	[  4.94279878e-04   2.03460070e+00]	[ -1.48508952e-02   1.06000000e+02]	[ -1.26938685e-02   1.21000000e+02]
    200	83    	[ -1.30314722e-02   1.12060000e+02]	[  5.24963259e-04   1.83204803e+00]	[ -1.47764152e-02   1.05000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    201	81    	[ -1.28869402e-02   1.12040000e+02]	[  3.78660672e-04   1.29553078e+00]	[ -1.46037187e-02   1.06000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    202	74    	[ -1.28750022e-02   1.12310000e+02]	[  3.82905786e-04   1.85846711e+00]	[ -1.41172970e-02   1.05000000e+02]	[ -1.26938685e-02   1.24000000e+02]
    203	72    	[ -1.28766226e-02   1.11850000e+02]	[  3.86564122e-04   1.26787223e+00]	[ -1.46406676e-02   1.06000000e+02]	[ -1.26938685e-02   1.15000000e+02]
    204	77    	[ -1.29188663e-02   1.12010000e+02]	[  4.28163793e-04   1.96211620e+00]	[ -1.42980717e-02   1.02000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    205	79    	[ -1.29666358e-02   1.12030000e+02]	[  5.02370805e-04   1.63373805e+00]	[ -1.50495384e-02   1.06000000e+02]	[ -1.26938685e-02   1.21000000e+02]
    206	73    	[ -1.28875425e-02   1.11780000e+02]	[  5.64419393e-04   1.47363496e+00]	[ -1.63413879e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    207	78    	[ -1.28610831e-02   1.11920000e+02]	[  4.72944408e-04   1.35410487e+00]	[ -1.48784706e-02   1.07000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    208	81    	[ -1.28271621e-02   1.12020000e+02]	[  3.67802636e-04   1.40698259e+00]	[ -1.47867769e-02   1.06000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    209	70    	[ -1.28798273e-02   1.11870000e+02]	[  4.28717595e-04   1.14590576e+00]	[ -1.48730900e-02   1.06000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    210	75    	[ -1.29531676e-02   1.12160000e+02]	[  5.23134028e-04   1.86397425e+00]	[ -1.49478263e-02   1.05000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    211	72    	[ -1.28553638e-02   1.12130000e+02]	[  3.81016919e-04   1.12831733e+00]	[ -1.44386431e-02   1.09000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    212	72    	[ -1.28479018e-02   1.11740000e+02]	[  3.87421256e-04   1.20515559e+00]	[ -1.47671179e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    213	77    	[ -1.28555267e-02   1.11680000e+02]	[  3.77103788e-04   1.32574507e+00]	[ -1.43323470e-02   1.06000000e+02]	[ -1.26938685e-02   1.15000000e+02]
    214	74    	[ -1.28297612e-02   1.12060000e+02]	[  3.33114675e-04   1.31011450e+00]	[ -1.42689693e-02   1.05000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    215	71    	[ -1.27972879e-02   1.12050000e+02]	[  3.41497484e-04   3.84057287e-01]	[ -1.48699891e-02   1.11000000e+02]	[ -1.26938685e-02   1.15000000e+02]
    216	76    	[ -1.28533691e-02   1.11960000e+02]	[  3.93834045e-04   1.23223374e+00]	[ -1.47482809e-02   1.04000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    217	68    	[ -1.28725528e-02   1.12110000e+02]	[  4.64341580e-04   1.65466009e+00]	[ -1.51654417e-02   1.05000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    218	80    	[ -1.29261002e-02   1.11840000e+02]	[  4.27759816e-04   1.40513345e+00]	[ -1.46591695e-02   1.05000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    219	76    	[ -1.28241750e-02   1.11990000e+02]	[  3.22139258e-04   1.58426639e+00]	[ -1.39442278e-02   1.05000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    220	71    	[ -1.28261510e-02   1.11900000e+02]	[  3.61047498e-04   9.74679434e-01]	[ -1.48786603e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    221	70    	[ -1.28726161e-02   1.11870000e+02]	[  3.62738174e-04   1.37589971e+00]	[ -1.42591953e-02   1.06000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    222	84    	[ -1.29266114e-02   1.11920000e+02]	[  4.96589534e-04   1.34669967e+00]	[ -1.49501136e-02   1.03000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    223	79    	[ -1.30205322e-02   1.12040000e+02]	[  5.41880991e-04   1.69068034e+00]	[ -1.58514092e-02   1.07000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    224	69    	[ -1.29133954e-02   1.12020000e+02]	[  4.94244601e-04   1.50319659e+00]	[ -1.57232127e-02   1.06000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    225	68    	[ -1.29426247e-02   1.11790000e+02]	[  5.01326332e-04   1.79050272e+00]	[ -1.58599674e-02   1.03000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    226	75    	[ -1.28833058e-02   1.12290000e+02]	[  4.84235254e-04   1.64496201e+00]	[ -1.60603290e-02   1.08000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    227	77    	[ -1.29275359e-02   1.12030000e+02]	[  4.72241103e-04   1.35244224e+00]	[ -1.50257738e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    228	69    	[ -1.29639142e-02   1.12090000e+02]	[  5.00926701e-04   2.11704983e+00]	[ -1.48690282e-02   1.02000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    229	74    	[ -1.28623885e-02   1.11930000e+02]	[  4.12658200e-04   1.29038754e+00]	[ -1.51976239e-02   1.06000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    230	80    	[ -1.28218995e-02   1.11850000e+02]	[  2.93076906e-04   1.25996032e+00]	[ -1.39069033e-02   1.08000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    231	75    	[ -1.29026007e-02   1.12090000e+02]	[  4.34333978e-04   1.42895066e+00]	[ -1.47923405e-02   1.07000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    232	75    	[ -1.29130318e-02   1.11960000e+02]	[  5.08019226e-04   1.10381158e+00]	[ -1.50475733e-02   1.06000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    233	71    	[ -1.28863076e-02   1.12070000e+02]	[  4.40885372e-04   1.07939798e+00]	[ -1.47616799e-02   1.08000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    234	80    	[ -1.28767391e-02   1.12090000e+02]	[  3.99701869e-04   1.09631200e+00]	[ -1.41633081e-02   1.08000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    235	69    	[ -1.28999709e-02   1.12080000e+02]	[  4.57151411e-04   1.15481600e+00]	[ -1.48381346e-02   1.09000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    236	78    	[ -1.29506695e-02   1.11950000e+02]	[  4.70818671e-04   1.76281026e+00]	[ -1.48485769e-02   1.05000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    237	69    	[ -1.29462479e-02   1.12020000e+02]	[  5.08714756e-04   1.09526253e+00]	[ -1.51981956e-02   1.08000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    238	79    	[ -1.28771693e-02   1.11770000e+02]	[  4.04800254e-04   1.39179740e+00]	[ -1.46362074e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    239	79    	[ -1.28972452e-02   1.11920000e+02]	[  4.08891859e-04   1.38332932e+00]	[ -1.44503409e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    240	81    	[ -1.29015839e-02   1.12120000e+02]	[  5.08125457e-04   1.38765990e+00]	[ -1.49797512e-02   1.07000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    241	76    	[ -1.29117839e-02   1.12220000e+02]	[  5.44030856e-04   1.23757828e+00]	[ -1.50088557e-02   1.07000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    242	87    	[ -1.28780234e-02   1.12050000e+02]	[  4.12745340e-04   1.63324830e+00]	[ -1.46710481e-02   1.03000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    243	79    	[ -1.28904853e-02   1.11850000e+02]	[  4.21486717e-04   1.44481833e+00]	[ -1.51610354e-02   1.07000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    244	68    	[ -1.29003717e-02   1.11930000e+02]	[  4.38070019e-04   1.85609806e+00]	[ -1.48314041e-02   1.04000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    245	86    	[ -1.28777149e-02   1.12040000e+02]	[  3.86927347e-04   1.29553078e+00]	[ -1.44165573e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    246	77    	[ -1.29562791e-02   1.12070000e+02]	[  5.51283257e-04   2.04575170e+00]	[ -1.53762013e-02   1.03000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    247	79    	[ -1.29049658e-02   1.11860000e+02]	[  4.08112944e-04   1.19180535e+00]	[ -1.47508216e-02   1.07000000e+02]	[ -1.26938685e-02   1.15000000e+02]
    248	71    	[ -1.29539564e-02   1.11920000e+02]	[  9.34192230e-04   1.00677704e+00]	[ -2.11435268e-02   1.09000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    249	71    	[ -1.28129297e-02   1.11860000e+02]	[  4.26006031e-04   1.21671689e+00]	[ -1.62590932e-02   1.06000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    250	79    	[ -1.29107224e-02   1.12140000e+02]	[  4.50207430e-04   1.49010067e+00]	[ -1.47120687e-02   1.07000000e+02]	[ -1.26938685e-02   1.22000000e+02]
    251	75    	[ -1.29195462e-02   1.11880000e+02]	[  4.96991949e-04   1.58921364e+00]	[ -1.50086842e-02   1.06000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    252	69    	[ -1.28937590e-02   1.11670000e+02]	[  5.28042856e-04   1.31190701e+00]	[ -1.57226869e-02   1.05000000e+02]	[ -1.26938685e-02   1.14000000e+02]
    253	65    	[ -1.28573366e-02   1.12110000e+02]	[  4.32343607e-04   1.10358507e+00]	[ -1.51566686e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    254	78    	[ -1.28230735e-02   1.11990000e+02]	[  3.49925621e-04   1.17042727e+00]	[ -1.48466646e-02   1.08000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    255	70    	[ -1.28255482e-02   1.12140000e+02]	[  3.18651495e-04   1.04899952e+00]	[ -1.39447804e-02   1.08000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    256	71    	[ -1.28499457e-02   1.11900000e+02]	[  4.20497642e-04   1.17046999e+00]	[ -1.49132330e-02   1.07000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    257	80    	[ -1.29519915e-02   1.12180000e+02]	[  4.57157506e-04   2.14186834e+00]	[ -1.44363565e-02   1.04000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    258	74    	[ -1.29374068e-02   1.11760000e+02]	[  5.07478501e-04   1.62554606e+00]	[ -1.52725357e-02   1.03000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    259	74    	[ -1.28926531e-02   1.12130000e+02]	[  4.32179600e-04   1.77569705e+00]	[ -1.45929202e-02   1.04000000e+02]	[ -1.26938685e-02   1.23000000e+02]
    260	74    	[ -1.28682088e-02   1.11940000e+02]	[  5.13448236e-04   1.35513837e+00]	[ -1.65314896e-02   1.08000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    261	72    	[ -1.29065325e-02   1.12100000e+02]	[  4.13313678e-04   1.41067360e+00]	[ -1.44915286e-02   1.04000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    262	85    	[ -1.29783030e-02   1.12120000e+02]	[  5.15650986e-04   2.15536540e+00]	[ -1.50490580e-02   1.00000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    263	71    	[ -1.29465647e-02   1.11930000e+02]	[  5.15785001e-04   1.79585634e+00]	[ -1.49457221e-02   1.04000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    264	83    	[ -1.28869444e-02   1.12240000e+02]	[  4.01460177e-04   1.61938260e+00]	[ -1.46575407e-02   1.04000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    265	73    	[ -1.29100184e-02   1.12020000e+02]	[  4.10043172e-04   1.77752637e+00]	[ -1.47352343e-02   1.04000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    266	61    	[ -1.28904575e-02   1.11930000e+02]	[  4.47451314e-04   1.58905632e+00]	[ -1.45870360e-02   1.03000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    267	75    	[ -1.28656749e-02   1.11870000e+02]	[  3.81772969e-04   1.45365058e+00]	[ -1.42224388e-02   1.04000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    268	73    	[ -1.28428891e-02   1.12150000e+02]	[  3.76403492e-04   1.54515371e+00]	[ -1.46252045e-02   1.07000000e+02]	[ -1.26938685e-02   1.21000000e+02]
    269	73    	[ -1.28488571e-02   1.12120000e+02]	[  3.65426410e-04   1.49184450e+00]	[ -1.45963709e-02   1.05000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    270	75    	[ -1.28709373e-02   1.12070000e+02]	[  4.20954155e-04   1.27479410e+00]	[ -1.50587077e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    271	82    	[ -1.29243843e-02   1.11920000e+02]	[  4.69932978e-04   2.05757138e+00]	[ -1.47081024e-02   1.03000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    272	66    	[ -1.29032159e-02   1.11790000e+02]	[  4.68521428e-04   1.72797569e+00]	[ -1.50305247e-02   1.04000000e+02]	[ -1.26938685e-02   1.15000000e+02]
    273	76    	[ -1.29093758e-02   1.11760000e+02]	[  4.47108302e-04   1.16721892e+00]	[ -1.46689661e-02   1.08000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    274	75    	[ -1.28680756e-02   1.11870000e+02]	[  4.40025208e-04   1.71845861e+00]	[ -1.54449657e-02   1.06000000e+02]	[ -1.26938685e-02   1.22000000e+02]
    275	78    	[ -1.28780385e-02   1.11830000e+02]	[  4.05606612e-04   1.51693770e+00]	[ -1.45363434e-02   1.06000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    276	80    	[ -1.28975332e-02   1.11890000e+02]	[  4.49313479e-04   1.42755035e+00]	[ -1.53163616e-02   1.05000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    277	70    	[ -1.28671437e-02   1.12150000e+02]	[  3.80290256e-04   1.51904575e+00]	[ -1.46676276e-02   1.07000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    278	71    	[ -1.29027446e-02   1.11830000e+02]	[  4.67092074e-04   1.53006536e+00]	[ -1.51336421e-02   1.06000000e+02]	[ -1.26938685e-02   1.21000000e+02]
    279	86    	[ -1.30341307e-02   1.12120000e+02]	[  6.10166761e-04   1.73366663e+00]	[ -1.65337962e-02   1.08000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    280	79    	[ -1.29984447e-02   1.11840000e+02]	[  5.04133284e-04   1.39799857e+00]	[ -1.48865484e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    281	83    	[ -1.29105801e-02   1.12000000e+02]	[  4.70585981e-04   1.38564065e+00]	[ -1.48759517e-02   1.07000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    282	72    	[ -1.28612805e-02   1.11990000e+02]	[  4.15160545e-04   9.43345112e-01]	[ -1.45449364e-02   1.08000000e+02]	[ -1.26938685e-02   1.15000000e+02]
    283	82    	[ -1.29117055e-02   1.11800000e+02]	[  4.83407591e-04   1.00995049e+00]	[ -1.50577616e-02   1.07000000e+02]	[ -1.26938685e-02   1.14000000e+02]
    284	77    	[ -1.29751074e-02   1.11570000e+02]	[  5.22908546e-04   1.77907279e+00]	[ -1.50489955e-02   1.05000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    285	80    	[ -1.29465307e-02   1.12080000e+02]	[  5.20473294e-04   1.48781719e+00]	[ -1.52182464e-02   1.06000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    286	84    	[ -1.29153620e-02   1.12010000e+02]	[  4.87345813e-04   1.55881365e+00]	[ -1.52415086e-02   1.05000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    287	73    	[ -1.28571574e-02   1.11950000e+02]	[  3.65267490e-04   1.16081868e+00]	[ -1.42837997e-02   1.08000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    288	74    	[ -1.29002616e-02   1.12130000e+02]	[  4.08360997e-04   1.45365058e+00]	[ -1.43897915e-02   1.06000000e+02]	[ -1.26938685e-02   1.20000000e+02]
    289	76    	[ -1.29227923e-02   1.11880000e+02]	[  5.69762234e-04   1.83455717e+00]	[ -1.53568245e-02   1.01000000e+02]	[ -1.26938685e-02   1.21000000e+02]
    290	81    	[ -1.28695248e-02   1.11940000e+02]	[  3.67004685e-04   1.26348724e+00]	[ -1.42572053e-02   1.05000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    291	74    	[ -1.28980332e-02   1.12050000e+02]	[  3.83015372e-04   1.08972474e+00]	[ -1.40327184e-02   1.08000000e+02]	[ -1.26938685e-02   1.17000000e+02]
    292	79    	[ -1.29236823e-02   1.12020000e+02]	[  4.76551826e-04   1.43513066e+00]	[ -1.46967566e-02   1.03000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    293	82    	[ -1.29492130e-02   1.11850000e+02]	[  4.78185424e-04   1.60234204e+00]	[ -1.50134482e-02   1.03000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    294	87    	[ -1.29651920e-02   1.11860000e+02]	[  4.58084039e-04   1.53635933e+00]	[ -1.47092772e-02   1.05000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    295	61    	[ -1.28252962e-02   1.12060000e+02]	[  3.27946298e-04   1.20681399e+00]	[ -1.40028764e-02   1.07000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    296	84    	[ -1.28996293e-02   1.11930000e+02]	[  4.80210803e-04   1.54437690e+00]	[ -1.50693930e-02   1.07000000e+02]	[ -1.26938685e-02   1.22000000e+02]
    297	80    	[ -1.29514211e-02   1.11860000e+02]	[  5.17399762e-04   1.51670696e+00]	[ -1.50949523e-02   1.04000000e+02]	[ -1.26938685e-02   1.16000000e+02]
    298	85    	[ -1.28661165e-02   1.11980000e+02]	[  3.44712340e-04   1.13119406e+00]	[ -1.40135795e-02   1.08000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    299	75    	[ -1.29292300e-02   1.12040000e+02]	[  4.60951907e-04   1.18253964e+00]	[ -1.45879200e-02   1.07000000e+02]	[ -1.26938685e-02   1.19000000e+02]
    300	72    	[ -1.29202628e-02   1.11890000e+02]	[  4.52620832e-04   1.78826732e+00]	[ -1.47256665e-02   1.04000000e+02]	[ -1.26938685e-02   1.18000000e+02]
    Selecting features with genetic algorithm.
    gen	nevals	avg                                	std                                	min                                	max                                
    0  	100   	[ -1.95531367e-02   1.08950000e+02]	[  3.09411640e-03   7.02335390e+00]	[ -3.63066380e-02   9.20000000e+01]	[ -1.57145499e-02   1.25000000e+02]
    1  	84    	[ -1.75104386e-02   1.10870000e+02]	[  1.14355945e-03   7.29747217e+00]	[ -2.22274763e-02   9.30000000e+01]	[ -1.54157174e-02   1.27000000e+02]
    2  	80    	[ -1.69493509e-02   1.10420000e+02]	[  1.24549015e-03   7.51023302e+00]	[ -2.36494445e-02   9.00000000e+01]	[ -1.50628612e-02   1.24000000e+02]
    3  	69    	[ -1.61499625e-02   1.11680000e+02]	[  9.15065015e-04   6.40918091e+00]	[ -2.11268315e-02   9.40000000e+01]	[ -1.47672799e-02   1.24000000e+02]
    4  	75    	[ -1.57451085e-02   1.11210000e+02]	[  7.35691746e-04   6.77834050e+00]	[ -1.83371447e-02   9.50000000e+01]	[ -1.43051715e-02   1.27000000e+02]
    5  	66    	[ -1.53081382e-02   1.12600000e+02]	[  5.26881571e-04   6.96993544e+00]	[ -1.66222949e-02   8.90000000e+01]	[ -1.42906226e-02   1.28000000e+02]
    6  	80    	[ -1.50613554e-02   1.13580000e+02]	[  6.65275515e-04   5.72394969e+00]	[ -1.85208597e-02   1.00000000e+02]	[ -1.38195617e-02   1.29000000e+02]
    7  	69    	[ -1.48489452e-02   1.14460000e+02]	[  6.14071903e-04   6.03062186e+00]	[ -1.81729678e-02   1.01000000e+02]	[ -1.39938618e-02   1.32000000e+02]
    8  	68    	[ -1.47677247e-02   1.14790000e+02]	[  8.64990861e-04   6.09802427e+00]	[ -2.06769369e-02   1.02000000e+02]	[ -1.35947334e-02   1.28000000e+02]
    9  	69    	[ -1.44617703e-02   1.15950000e+02]	[  3.90623916e-04   6.05371787e+00]	[ -1.62510870e-02   9.80000000e+01]	[ -1.35789480e-02   1.29000000e+02]
    10 	78    	[ -1.43875896e-02   1.14860000e+02]	[  4.24307149e-04   4.87241213e+00]	[ -1.66779604e-02   1.04000000e+02]	[ -1.35103563e-02   1.29000000e+02]
    11 	69    	[ -1.42305762e-02   1.14590000e+02]	[  3.37904188e-04   5.59838370e+00]	[ -1.55823552e-02   1.03000000e+02]	[ -1.35838656e-02   1.30000000e+02]
    12 	72    	[ -1.41054967e-02   1.13710000e+02]	[  3.31697923e-04   5.94019360e+00]	[ -1.51706360e-02   1.01000000e+02]	[ -1.34984401e-02   1.37000000e+02]
    13 	81    	[ -1.40430286e-02   1.13320000e+02]	[  5.76467384e-04   5.17470772e+00]	[ -1.84437467e-02   1.01000000e+02]	[ -1.33582085e-02   1.28000000e+02]
    14 	87    	[ -1.38868124e-02   1.14040000e+02]	[  6.36449051e-04   5.53880854e+00]	[ -1.94394879e-02   1.00000000e+02]	[ -1.32659186e-02   1.26000000e+02]
    15 	83    	[ -1.37465363e-02   1.14160000e+02]	[  4.09055125e-04   5.65105300e+00]	[ -1.65416243e-02   1.03000000e+02]	[ -1.30090777e-02   1.30000000e+02]
    16 	66    	[ -1.35743478e-02   1.13180000e+02]	[  2.87996802e-04   4.90587403e+00]	[ -1.46827570e-02   1.04000000e+02]	[ -1.30218557e-02   1.30000000e+02]
    17 	76    	[ -1.35356223e-02   1.12760000e+02]	[  4.41487112e-04   5.25189490e+00]	[ -1.60473099e-02   1.00000000e+02]	[ -1.27641272e-02   1.25000000e+02]
    18 	81    	[ -1.34458733e-02   1.12440000e+02]	[  4.53239320e-04   5.09572370e+00]	[ -1.60206736e-02   1.01000000e+02]	[ -1.27641272e-02   1.26000000e+02]
    19 	67    	[ -1.33006323e-02   1.11330000e+02]	[  4.54716987e-04   5.01409015e+00]	[ -1.62257458e-02   1.00000000e+02]	[ -1.27641272e-02   1.24000000e+02]
    20 	72    	[ -1.31500052e-02   1.11710000e+02]	[  2.83729598e-04   4.63960128e+00]	[ -1.44700227e-02   1.00000000e+02]	[ -1.27045523e-02   1.21000000e+02]
    21 	68    	[ -1.30312907e-02   1.13050000e+02]	[  2.20558673e-04   4.22699657e+00]	[ -1.39903975e-02   1.01000000e+02]	[ -1.26811369e-02   1.22000000e+02]
    22 	76    	[ -1.29388215e-02   1.13590000e+02]	[  2.23281158e-04   3.62517586e+00]	[ -1.39935487e-02   1.04000000e+02]	[ -1.25514759e-02   1.23000000e+02]
    23 	80    	[ -1.30073175e-02   1.13110000e+02]	[  5.26178425e-04   4.37468856e+00]	[ -1.55510653e-02   1.02000000e+02]	[ -1.25326854e-02   1.22000000e+02]
    24 	74    	[ -1.29056187e-02   1.13070000e+02]	[  4.50502523e-04   4.66745112e+00]	[ -1.51747062e-02   9.70000000e+01]	[ -1.24759943e-02   1.23000000e+02]
    25 	65    	[ -1.27671020e-02   1.13420000e+02]	[  2.71830896e-04   4.32707754e+00]	[ -1.42830982e-02   1.04000000e+02]	[ -1.23643769e-02   1.26000000e+02]
    26 	73    	[ -1.27956553e-02   1.12770000e+02]	[  4.74317872e-04   4.31707077e+00]	[ -1.50442024e-02   1.00000000e+02]	[ -1.23549946e-02   1.23000000e+02]
    27 	74    	[ -1.27392827e-02   1.11360000e+02]	[  5.41390283e-04   3.70275573e+00]	[ -1.57634904e-02   1.04000000e+02]	[ -1.23549946e-02   1.20000000e+02]
    28 	83    	[ -1.27344002e-02   1.10720000e+02]	[  7.16726429e-04   3.69345367e+00]	[ -1.82345556e-02   1.02000000e+02]	[ -1.23394375e-02   1.18000000e+02]
    29 	73    	[ -1.26191946e-02   1.10420000e+02]	[  6.67968754e-04   3.46462119e+00]	[ -1.88377952e-02   1.02000000e+02]	[ -1.22054248e-02   1.18000000e+02]
    30 	72    	[ -1.25586854e-02   1.10220000e+02]	[  2.74632767e-04   3.29721094e+00]	[ -1.38783864e-02   1.04000000e+02]	[ -1.21975114e-02   1.19000000e+02]
    31 	66    	[ -1.25418305e-02   1.09620000e+02]	[  5.53679811e-04   3.41988304e+00]	[ -1.58762644e-02   1.02000000e+02]	[ -1.21975114e-02   1.19000000e+02]
    32 	74    	[ -1.24615009e-02   1.09710000e+02]	[  4.02540074e-04   3.78495707e+00]	[ -1.48501789e-02   1.03000000e+02]	[ -1.21929122e-02   1.18000000e+02]
    33 	69    	[ -1.24078571e-02   1.09340000e+02]	[  2.84724818e-04   4.25727612e+00]	[ -1.37692182e-02   1.02000000e+02]	[ -1.21488181e-02   1.21000000e+02]
    34 	76    	[ -1.23715987e-02   1.08300000e+02]	[  2.95167332e-04   3.89743505e+00]	[ -1.38446088e-02   9.40000000e+01]	[ -1.21454787e-02   1.17000000e+02]
    35 	72    	[ -1.23738816e-02   1.07690000e+02]	[  3.83981413e-04   3.65703432e+00]	[ -1.49793246e-02   1.00000000e+02]	[ -1.20909551e-02   1.19000000e+02]
    36 	75    	[ -1.22989042e-02   1.07260000e+02]	[  3.83241752e-04   3.19881228e+00]	[ -1.54456155e-02   1.00000000e+02]	[ -1.20661224e-02   1.15000000e+02]
    37 	76    	[ -1.22841191e-02   1.07180000e+02]	[  5.57032870e-04   3.28140214e+00]	[ -1.74771471e-02   1.00000000e+02]	[ -1.20495704e-02   1.16000000e+02]
    38 	77    	[ -1.22445069e-02   1.06290000e+02]	[  2.97130731e-04   3.17897782e+00]	[ -1.39937867e-02   9.50000000e+01]	[ -1.20248306e-02   1.14000000e+02]
    39 	76    	[ -1.23575759e-02   1.05640000e+02]	[  7.29653903e-04   2.79828519e+00]	[ -1.76260109e-02   9.70000000e+01]	[ -1.20248306e-02   1.12000000e+02]
    40 	79    	[ -1.22255847e-02   1.06250000e+02]	[  3.69477839e-04   2.66973781e+00]	[ -1.44286872e-02   9.90000000e+01]	[ -1.20264955e-02   1.16000000e+02]
    41 	73    	[ -1.24137667e-02   1.05930000e+02]	[  1.15915242e-03   2.53477810e+00]	[ -1.85607752e-02   9.80000000e+01]	[ -1.20177461e-02   1.13000000e+02]
    42 	75    	[ -1.22998872e-02   1.05610000e+02]	[  8.42685065e-04   2.32333812e+00]	[ -1.81551722e-02   1.00000000e+02]	[ -1.20177461e-02   1.12000000e+02]
    43 	76    	[ -1.22233296e-02   1.05940000e+02]	[  6.56661984e-04   2.09198470e+00]	[ -1.72654797e-02   1.01000000e+02]	[ -1.20046918e-02   1.11000000e+02]
    44 	84    	[ -1.22208396e-02   1.05250000e+02]	[  7.12570557e-04   2.07062792e+00]	[ -1.77100884e-02   9.90000000e+01]	[ -1.19898807e-02   1.11000000e+02]
    45 	76    	[ -1.21657469e-02   1.04970000e+02]	[  3.39512836e-04   2.24702025e+00]	[ -1.41680768e-02   9.80000000e+01]	[ -1.19908647e-02   1.13000000e+02]
    46 	77    	[ -1.22143162e-02   1.04110000e+02]	[  5.65594969e-04   2.26227761e+00]	[ -1.60937770e-02   9.90000000e+01]	[ -1.19896803e-02   1.12000000e+02]
    47 	77    	[ -1.22398320e-02   1.03600000e+02]	[  8.96214819e-04   2.16333077e+00]	[ -1.83432511e-02   9.50000000e+01]	[ -1.19873854e-02   1.12000000e+02]
    48 	85    	[ -1.20723051e-02   1.03410000e+02]	[  2.04498591e-04   1.90312900e+00]	[ -1.31248392e-02   9.90000000e+01]	[ -1.19866568e-02   1.12000000e+02]
    49 	78    	[ -1.22367783e-02   1.03170000e+02]	[  8.54680477e-04   1.80030553e+00]	[ -1.74601778e-02   1.00000000e+02]	[ -1.19780403e-02   1.13000000e+02]
    50 	67    	[ -1.21530704e-02   1.02800000e+02]	[  5.96375755e-04   1.94422221e+00]	[ -1.74540763e-02   9.90000000e+01]	[ -1.19777571e-02   1.12000000e+02]
    51 	71    	[ -1.21675466e-02   1.02090000e+02]	[  6.97534375e-04   1.56904430e+00]	[ -1.84807809e-02   9.70000000e+01]	[ -1.19682670e-02   1.07000000e+02]
    52 	68    	[ -1.21784354e-02   1.01760000e+02]	[  4.77838803e-04   2.16850179e+00]	[ -1.48030498e-02   9.40000000e+01]	[ -1.19533354e-02   1.11000000e+02]
    53 	71    	[ -1.21008184e-02   1.01710000e+02]	[  3.43684684e-04   1.53163312e+00]	[ -1.40489107e-02   9.90000000e+01]	[ -1.19510005e-02   1.07000000e+02]
    54 	63    	[ -1.21398645e-02   1.01040000e+02]	[  5.76275793e-04   1.77718879e+00]	[ -1.57029996e-02   9.10000000e+01]	[ -1.19495561e-02   1.05000000e+02]
    55 	86    	[ -1.21267077e-02   1.01170000e+02]	[  3.46341458e-04   2.27620298e+00]	[ -1.34037278e-02   9.50000000e+01]	[ -1.19461454e-02   1.09000000e+02]
    56 	90    	[ -1.20936061e-02   1.01020000e+02]	[  3.99851090e-04   2.08796552e+00]	[ -1.40387989e-02   9.70000000e+01]	[ -1.19458432e-02   1.08000000e+02]
    57 	78    	[ -1.20089313e-02   1.00560000e+02]	[  2.11101765e-04   1.59574434e+00]	[ -1.36475205e-02   9.70000000e+01]	[ -1.19458432e-02   1.07000000e+02]
    58 	71    	[ -1.21424283e-02   1.00390000e+02]	[  6.35508921e-04   1.76575763e+00]	[ -1.73435057e-02   9.40000000e+01]	[ -1.19060691e-02   1.06000000e+02]
    59 	77    	[ -1.21476788e-02   9.97000000e+01]	[  6.20060283e-04   1.92613603e+00]	[ -1.66725680e-02   9.30000000e+01]	[ -1.19088800e-02   1.06000000e+02]
    60 	69    	[ -1.21147017e-02   9.90500000e+01]	[  4.29380804e-04   1.87283208e+00]	[ -1.37275857e-02   9.10000000e+01]	[ -1.19053769e-02   1.07000000e+02]
    61 	79    	[ -1.21766745e-02   9.89500000e+01]	[  6.76260864e-04   2.44284670e+00]	[ -1.78263974e-02   9.30000000e+01]	[ -1.19053769e-02   1.08000000e+02]
    62 	82    	[ -1.21270164e-02   9.79500000e+01]	[  3.68609106e-04   2.32109888e+00]	[ -1.34843933e-02   9.10000000e+01]	[ -1.18763570e-02   1.07000000e+02]
    63 	79    	[ -1.21818678e-02   9.74100000e+01]	[  7.17046359e-04   2.08851622e+00]	[ -1.64040177e-02   9.20000000e+01]	[ -1.18740889e-02   1.04000000e+02]
    64 	79    	[ -1.21403155e-02   9.76400000e+01]	[  5.49188204e-04   2.35168025e+00]	[ -1.58529986e-02   9.30000000e+01]	[ -1.18692957e-02   1.05000000e+02]
    65 	88    	[ -1.20729001e-02   9.68600000e+01]	[  6.00384877e-04   2.14951157e+00]	[ -1.72396719e-02   9.10000000e+01]	[ -1.18508015e-02   1.05000000e+02]
    66 	73    	[ -1.20644255e-02   9.64000000e+01]	[  7.60684549e-04   2.10237960e+00]	[ -1.79857826e-02   9.10000000e+01]	[ -1.18508015e-02   1.04000000e+02]
    67 	74    	[ -1.20278076e-02   9.55700000e+01]	[  6.04931327e-04   1.93522608e+00]	[ -1.74021665e-02   8.80000000e+01]	[ -1.18508015e-02   1.02000000e+02]
    68 	75    	[ -1.20162350e-02   9.48900000e+01]	[  3.45544271e-04   2.39956246e+00]	[ -1.38236406e-02   8.70000000e+01]	[ -1.18508015e-02   1.08000000e+02]
    69 	69    	[ -1.21102383e-02   9.43300000e+01]	[  9.41856858e-04   1.98522039e+00]	[ -2.06771978e-02   8.90000000e+01]	[ -1.18508015e-02   1.02000000e+02]
    70 	72    	[ -1.20779446e-02   9.38200000e+01]	[  6.30885888e-04   1.60237324e+00]	[ -1.71774373e-02   9.00000000e+01]	[ -1.18508015e-02   1.00000000e+02]
    71 	80    	[ -1.19152105e-02   9.32700000e+01]	[  1.97940849e-04   1.52220235e+00]	[ -1.32870715e-02   8.90000000e+01]	[ -1.18363053e-02   9.90000000e+01]
    72 	70    	[ -1.21499988e-02   9.27800000e+01]	[  1.01079197e-03   1.67678263e+00]	[ -1.80897859e-02   8.80000000e+01]	[ -1.18363053e-02   1.00000000e+02]
    73 	76    	[ -1.20168918e-02   9.25200000e+01]	[  4.17979317e-04   1.45931491e+00]	[ -1.39252201e-02   8.80000000e+01]	[ -1.18348199e-02   1.00000000e+02]
    74 	70    	[ -1.20152803e-02   9.22800000e+01]	[  6.43854125e-04   1.87125626e+00]	[ -1.68857643e-02   8.50000000e+01]	[ -1.18348199e-02   1.01000000e+02]
    75 	78    	[ -1.21571187e-02   9.17000000e+01]	[  1.10752120e-03   1.20415946e+00]	[ -1.81916272e-02   8.70000000e+01]	[ -1.18348199e-02   9.70000000e+01]
    76 	77    	[ -1.19756853e-02   9.18300000e+01]	[  4.61054362e-04   1.87112266e+00]	[ -1.60513255e-02   8.70000000e+01]	[ -1.18348199e-02   9.90000000e+01]
    77 	78    	[ -1.21552134e-02   9.14800000e+01]	[  1.26700232e-03   1.68214149e+00]	[ -2.11535146e-02   8.60000000e+01]	[ -1.18348199e-02   1.00000000e+02]
    78 	76    	[ -1.19955552e-02   9.15500000e+01]	[  4.05670435e-04   1.71682847e+00]	[ -1.38708143e-02   8.80000000e+01]	[ -1.18348199e-02   9.90000000e+01]
    79 	86    	[ -1.19965494e-02   9.14800000e+01]	[  4.02550785e-04   1.67020957e+00]	[ -1.39816175e-02   8.60000000e+01]	[ -1.18348199e-02   9.90000000e+01]
    80 	84    	[ -1.19849612e-02   9.13200000e+01]	[  3.79989404e-04   1.25602548e+00]	[ -1.35249118e-02   8.70000000e+01]	[ -1.18348199e-02   9.60000000e+01]
    81 	65    	[ -1.20770030e-02   9.12600000e+01]	[  7.72041522e-04   1.59135163e+00]	[ -1.79480273e-02   8.60000000e+01]	[ -1.18238841e-02   1.01000000e+02]
    82 	76    	[ -1.20518135e-02   9.16500000e+01]	[  5.83388841e-04   1.82961745e+00]	[ -1.50087730e-02   8.70000000e+01]	[ -1.18231761e-02   9.90000000e+01]
    83 	76    	[ -1.20617055e-02   9.13300000e+01]	[  7.61525915e-04   1.92382432e+00]	[ -1.85710977e-02   8.60000000e+01]	[ -1.18172381e-02   1.00000000e+02]
    84 	72    	[ -1.20697502e-02   9.10700000e+01]	[  8.89074628e-04   1.43704558e+00]	[ -1.80927884e-02   8.40000000e+01]	[ -1.18172381e-02   9.70000000e+01]
    85 	79    	[ -1.19933966e-02   9.15800000e+01]	[  3.93559218e-04   1.81207064e+00]	[ -1.45283250e-02   8.80000000e+01]	[ -1.18172381e-02   1.00000000e+02]
    86 	78    	[ -1.20700331e-02   9.13400000e+01]	[  7.09162334e-04   1.35808689e+00]	[ -1.72074918e-02   8.60000000e+01]	[ -1.18172381e-02   9.70000000e+01]
    87 	71    	[ -1.19165046e-02   9.17200000e+01]	[  2.63857139e-04   1.63755916e+00]	[ -1.31430587e-02   8.80000000e+01]	[ -1.18172381e-02   1.00000000e+02]
    88 	73    	[ -1.19764037e-02   9.20900000e+01]	[  3.83891625e-04   2.02531479e+00]	[ -1.35565844e-02   8.70000000e+01]	[ -1.18172381e-02   1.02000000e+02]
    89 	79    	[ -1.19272646e-02   9.20800000e+01]	[  3.34165892e-04   1.70692706e+00]	[ -1.39342800e-02   8.50000000e+01]	[ -1.18003378e-02   1.00000000e+02]
    90 	74    	[ -1.20226473e-02   9.26200000e+01]	[  6.90596769e-04   1.86429611e+00]	[ -1.80681844e-02   8.40000000e+01]	[ -1.17723663e-02   1.00000000e+02]
    91 	76    	[ -1.20355695e-02   9.25900000e+01]	[  4.87904943e-04   1.77817322e+00]	[ -1.40276530e-02   8.90000000e+01]	[ -1.17656992e-02   1.01000000e+02]
    92 	76    	[ -1.20790401e-02   9.25700000e+01]	[  8.34177471e-04   1.54437690e+00]	[ -1.73731585e-02   8.90000000e+01]	[ -1.17656992e-02   9.90000000e+01]
    93 	79    	[ -1.20088874e-02   9.24400000e+01]	[  4.71748988e-04   1.86182706e+00]	[ -1.46061063e-02   9.00000000e+01]	[ -1.17656992e-02   1.01000000e+02]
    94 	64    	[ -1.19359498e-02   9.20600000e+01]	[  6.27577611e-04   1.52852870e+00]	[ -1.74318255e-02   8.60000000e+01]	[ -1.17656992e-02   9.70000000e+01]
    95 	84    	[ -1.20010079e-02   9.23100000e+01]	[  8.68262110e-04   1.74754113e+00]	[ -1.98953808e-02   8.90000000e+01]	[ -1.17656992e-02   1.02000000e+02]
    96 	82    	[ -1.18598472e-02   9.22400000e+01]	[  4.02899052e-04   1.40797727e+00]	[ -1.54378753e-02   8.80000000e+01]	[ -1.17656992e-02   1.00000000e+02]
    97 	73    	[ -1.19422566e-02   9.22700000e+01]	[  3.98987629e-04   1.83223907e+00]	[ -1.37012680e-02   8.60000000e+01]	[ -1.17656992e-02   1.02000000e+02]
    98 	68    	[ -1.18868144e-02   9.23100000e+01]	[  3.23041714e-04   1.44703144e+00]	[ -1.34394148e-02   8.80000000e+01]	[ -1.17339679e-02   9.90000000e+01]
    99 	87    	[ -1.19222370e-02   9.26300000e+01]	[  3.27944174e-04   2.48054430e+00]	[ -1.31681401e-02   8.60000000e+01]	[ -1.17310071e-02   1.05000000e+02]
    100	76    	[ -1.18737537e-02   9.21700000e+01]	[  3.21358701e-04   1.37880383e+00]	[ -1.37125873e-02   9.00000000e+01]	[ -1.17310071e-02   9.80000000e+01]
    101	68    	[ -1.19229191e-02   9.20200000e+01]	[  6.55381175e-04   1.28046866e+00]	[ -1.76260596e-02   9.00000000e+01]	[ -1.17310071e-02   9.70000000e+01]
    102	78    	[ -1.19670888e-02   9.15400000e+01]	[  8.78397842e-04   1.53896069e+00]	[ -1.80401540e-02   8.80000000e+01]	[ -1.17310071e-02   9.90000000e+01]
    103	60    	[ -1.18679522e-02   9.15000000e+01]	[  3.30420190e-04   2.00249844e+00]	[ -1.40452487e-02   8.30000000e+01]	[ -1.17310071e-02   1.02000000e+02]
    104	81    	[ -1.18933638e-02   9.13800000e+01]	[  3.92350382e-04   2.32284309e+00]	[ -1.37566523e-02   8.60000000e+01]	[ -1.17310071e-02   1.02000000e+02]
    105	76    	[ -1.20344403e-02   9.09400000e+01]	[  9.71926038e-04   1.89641768e+00]	[ -1.83781103e-02   8.80000000e+01]	[ -1.17310071e-02   1.01000000e+02]
    106	63    	[ -1.19258474e-02   9.04600000e+01]	[  6.88423721e-04   1.65178691e+00]	[ -1.78684178e-02   8.50000000e+01]	[ -1.17310071e-02   9.80000000e+01]
    107	78    	[ -1.18449892e-02   9.02200000e+01]	[  3.15146291e-04   1.11874930e+00]	[ -1.33208713e-02   8.70000000e+01]	[ -1.17310071e-02   9.50000000e+01]
    108	79    	[ -1.19032723e-02   9.03200000e+01]	[  6.61995567e-04   1.92291445e+00]	[ -1.76566543e-02   8.40000000e+01]	[ -1.17310071e-02   9.80000000e+01]
    109	86    	[ -1.18672624e-02   9.05000000e+01]	[  3.10252260e-04   2.08086520e+00]	[ -1.31885043e-02   8.50000000e+01]	[ -1.17310071e-02   1.02000000e+02]
    110	82    	[ -1.19389333e-02   9.02700000e+01]	[  7.19853986e-04   1.56751396e+00]	[ -1.81356643e-02   8.50000000e+01]	[ -1.17310071e-02   9.80000000e+01]
    111	71    	[ -1.20301572e-02   9.03600000e+01]	[  1.08425514e-03   1.79733136e+00]	[ -1.87166491e-02   8.50000000e+01]	[ -1.17310071e-02   9.70000000e+01]
    112	81    	[ -1.19758778e-02   9.04000000e+01]	[  7.29180517e-04   1.77763888e+00]	[ -1.75106370e-02   8.50000000e+01]	[ -1.17310071e-02   9.70000000e+01]
    113	80    	[ -1.19889989e-02   9.04200000e+01]	[  9.74010590e-04   2.24133889e+00]	[ -2.04200876e-02   8.10000000e+01]	[ -1.17310071e-02   1.01000000e+02]
    114	70    	[ -1.20020625e-02   9.07200000e+01]	[  8.87371689e-04   2.04978048e+00]	[ -1.78732689e-02   8.50000000e+01]	[ -1.17310071e-02   9.90000000e+01]
    115	69    	[ -1.18657638e-02   9.02900000e+01]	[  3.37527673e-04   1.47169970e+00]	[ -1.34593211e-02   8.50000000e+01]	[ -1.17310071e-02   9.80000000e+01]
    116	77    	[ -1.19278897e-02   9.03100000e+01]	[  6.25785450e-04   1.54722332e+00]	[ -1.72358635e-02   8.40000000e+01]	[ -1.16406841e-02   9.80000000e+01]
    117	73    	[ -1.19387134e-02   9.00900000e+01]	[  7.45984563e-04   1.28136646e+00]	[ -1.83968455e-02   8.60000000e+01]	[ -1.16406841e-02   9.50000000e+01]
    118	70    	[ -1.19150416e-02   9.00400000e+01]	[  8.06048007e-04   1.48270024e+00]	[ -1.78514154e-02   8.50000000e+01]	[ -1.16406841e-02   9.90000000e+01]
    119	80    	[ -1.19866700e-02   8.99300000e+01]	[  1.20742536e-03   1.46461599e+00]	[ -2.14734851e-02   8.60000000e+01]	[ -1.16406841e-02   9.70000000e+01]
    120	72    	[ -1.19971283e-02   8.97500000e+01]	[  8.91396533e-04   2.28637267e+00]	[ -1.90809647e-02   8.40000000e+01]	[ -1.16406841e-02   9.80000000e+01]
    121	74    	[ -1.19056422e-02   8.92400000e+01]	[  8.64429727e-04   1.92416216e+00]	[ -1.98845134e-02   8.10000000e+01]	[ -1.16406841e-02   9.60000000e+01]
    122	83    	[ -1.20205919e-02   8.89600000e+01]	[  8.33170488e-04   2.34486673e+00]	[ -1.88487923e-02   8.30000000e+01]	[ -1.16406841e-02   9.60000000e+01]
    123	81    	[ -1.20395917e-02   8.81200000e+01]	[  1.06640115e-03   1.97625909e+00]	[ -1.95541827e-02   8.30000000e+01]	[ -1.16406841e-02   9.60000000e+01]
    124	72    	[ -1.18705984e-02   8.75600000e+01]	[  7.55354208e-04   1.69894085e+00]	[ -1.72349786e-02   8.50000000e+01]	[ -1.16406841e-02   9.50000000e+01]
    125	76    	[ -1.18556061e-02   8.72300000e+01]	[  5.79734217e-04   1.78244214e+00]	[ -1.59891459e-02   8.40000000e+01]	[ -1.16406841e-02   9.60000000e+01]
    126	77    	[ -1.17946527e-02   8.65700000e+01]	[  5.91159865e-04   1.55083848e+00]	[ -1.70645786e-02   8.10000000e+01]	[ -1.16406841e-02   9.50000000e+01]
    127	75    	[ -1.17573185e-02   8.68100000e+01]	[  2.62640184e-04   1.86384012e+00]	[ -1.30793998e-02   8.50000000e+01]	[ -1.16406841e-02   9.60000000e+01]
    128	81    	[ -1.19631275e-02   8.62300000e+01]	[  9.34174614e-04   1.57388055e+00]	[ -1.96640151e-02   8.20000000e+01]	[ -1.16406841e-02   9.20000000e+01]
    129	76    	[ -1.17855068e-02   8.63000000e+01]	[  5.17150794e-04   1.20415946e+00]	[ -1.58449895e-02   8.40000000e+01]	[ -1.16406841e-02   9.30000000e+01]
    130	79    	[ -1.18582344e-02   8.65900000e+01]	[  6.60797742e-04   1.84442403e+00]	[ -1.72982739e-02   8.30000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    131	81    	[ -1.17441956e-02   8.64100000e+01]	[  2.44139901e-04   1.34234869e+00]	[ -1.30215772e-02   8.50000000e+01]	[ -1.16406841e-02   9.30000000e+01]
    132	71    	[ -1.18259219e-02   8.65500000e+01]	[  4.97216775e-04   1.65151446e+00]	[ -1.54128494e-02   8.20000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    133	77    	[ -1.18272379e-02   8.66500000e+01]	[  4.12534605e-04   1.94100489e+00]	[ -1.40833743e-02   8.20000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    134	70    	[ -1.18074120e-02   8.62000000e+01]	[  5.01404514e-04   1.32664992e+00]	[ -1.49294230e-02   8.30000000e+01]	[ -1.16406841e-02   9.20000000e+01]
    135	82    	[ -1.17194730e-02   8.64000000e+01]	[  2.54436409e-04   1.62480768e+00]	[ -1.34077872e-02   8.40000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    136	83    	[ -1.17644569e-02   8.63500000e+01]	[  3.18838978e-04   1.36656504e+00]	[ -1.36097750e-02   8.30000000e+01]	[ -1.16406841e-02   9.30000000e+01]
    137	76    	[ -1.17492352e-02   8.63000000e+01]	[  3.30566480e-04   1.04403065e+00]	[ -1.31436048e-02   8.40000000e+01]	[ -1.16406841e-02   9.10000000e+01]
    138	69    	[ -1.18876106e-02   8.63300000e+01]	[  8.11917691e-04   1.25741799e+00]	[ -1.86622975e-02   8.30000000e+01]	[ -1.16406841e-02   9.20000000e+01]
    139	77    	[ -1.17913794e-02   8.62200000e+01]	[  3.87216703e-04   1.53349275e+00]	[ -1.36238637e-02   8.10000000e+01]	[ -1.16406841e-02   9.20000000e+01]
    140	75    	[ -1.19720506e-02   8.66200000e+01]	[  1.10993507e-03   1.54777259e+00]	[ -1.88593824e-02   8.40000000e+01]	[ -1.16406841e-02   9.20000000e+01]
    141	75    	[ -1.18093288e-02   8.65100000e+01]	[  7.13392159e-04   1.92611007e+00]	[ -1.82853284e-02   8.10000000e+01]	[ -1.16406841e-02   9.60000000e+01]
    142	74    	[ -1.18838889e-02   8.64700000e+01]	[  6.47922045e-04   1.81909318e+00]	[ -1.55586246e-02   8.20000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    143	89    	[ -1.18569639e-02   8.69200000e+01]	[  4.17269107e-04   2.31810267e+00]	[ -1.34211154e-02   8.40000000e+01]	[ -1.16406841e-02   1.00000000e+02]
    144	81    	[ -1.17948189e-02   8.63800000e+01]	[  3.71796940e-04   1.54777259e+00]	[ -1.35064316e-02   8.30000000e+01]	[ -1.16406841e-02   9.60000000e+01]
    145	76    	[ -1.17370769e-02   8.65900000e+01]	[  2.66442809e-04   1.68579358e+00]	[ -1.30781211e-02   8.50000000e+01]	[ -1.16406841e-02   9.50000000e+01]
    146	78    	[ -1.19151483e-02   8.66300000e+01]	[  6.84734839e-04   2.00327232e+00]	[ -1.68271674e-02   8.00000000e+01]	[ -1.16406841e-02   9.50000000e+01]
    147	75    	[ -1.17680619e-02   8.63200000e+01]	[  4.07225591e-04   1.24000000e+00]	[ -1.42538759e-02   8.30000000e+01]	[ -1.16406841e-02   9.20000000e+01]
    148	81    	[ -1.18705918e-02   8.67600000e+01]	[  6.69929894e-04   2.13129069e+00]	[ -1.74075378e-02   8.20000000e+01]	[ -1.16406841e-02   9.50000000e+01]
    149	78    	[ -1.19050464e-02   8.66600000e+01]	[  9.98892106e-04   2.01603571e+00]	[ -1.94469471e-02   8.20000000e+01]	[ -1.16406841e-02   9.80000000e+01]
    150	76    	[ -1.17516678e-02   8.64500000e+01]	[  3.11973718e-04   1.60857079e+00]	[ -1.30995724e-02   8.30000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    151	73    	[ -1.18821731e-02   8.62600000e+01]	[  9.36802987e-04   1.22979673e+00]	[ -1.89343881e-02   8.20000000e+01]	[ -1.16406841e-02   9.30000000e+01]
    152	74    	[ -1.17664743e-02   8.64700000e+01]	[  2.74445344e-04   1.51297720e+00]	[ -1.29822941e-02   8.30000000e+01]	[ -1.16406841e-02   9.30000000e+01]
    153	83    	[ -1.17809477e-02   8.65600000e+01]	[  3.13823052e-04   2.13222888e+00]	[ -1.31421223e-02   7.90000000e+01]	[ -1.16406841e-02   9.60000000e+01]
    154	82    	[ -1.17467044e-02   8.65400000e+01]	[  2.55003188e-04   1.84076071e+00]	[ -1.29439822e-02   8.30000000e+01]	[ -1.16406841e-02   9.70000000e+01]
    155	79    	[ -1.19000173e-02   8.66800000e+01]	[  7.02281398e-04   2.02425295e+00]	[ -1.72685521e-02   8.30000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    156	84    	[ -1.17492459e-02   8.65300000e+01]	[  2.90619350e-04   1.81909318e+00]	[ -1.36501703e-02   8.10000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    157	79    	[ -1.19630240e-02   8.69100000e+01]	[  1.08844571e-03   2.26757580e+00]	[ -1.91188769e-02   8.20000000e+01]	[ -1.16406841e-02   9.70000000e+01]
    158	66    	[ -1.18391655e-02   8.66200000e+01]	[  6.42954380e-04   1.58606431e+00]	[ -1.68990951e-02   8.40000000e+01]	[ -1.16406841e-02   9.30000000e+01]
    159	83    	[ -1.18555600e-02   8.64900000e+01]	[  7.55213558e-04   1.83572874e+00]	[ -1.85400064e-02   8.10000000e+01]	[ -1.16406841e-02   9.50000000e+01]
    160	72    	[ -1.18576810e-02   8.66700000e+01]	[  9.28514462e-04   1.78356385e+00]	[ -2.04384282e-02   8.40000000e+01]	[ -1.16406841e-02   9.50000000e+01]
    161	77    	[ -1.18790020e-02   8.64500000e+01]	[  6.65273865e-04   1.32947358e+00]	[ -1.70924420e-02   8.40000000e+01]	[ -1.16406841e-02   9.30000000e+01]
    162	79    	[ -1.17610070e-02   8.62400000e+01]	[  2.99938023e-04   1.00119928e+00]	[ -1.32793145e-02   8.40000000e+01]	[ -1.16406841e-02   9.00000000e+01]
    163	75    	[ -1.19048374e-02   8.66300000e+01]	[  7.73103062e-04   1.86898368e+00]	[ -1.78763284e-02   8.30000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    164	69    	[ -1.18304086e-02   8.64700000e+01]	[  7.06298637e-04   1.74043098e+00]	[ -1.79188018e-02   8.00000000e+01]	[ -1.16406841e-02   9.40000000e+01]
    165	82    	[ -1.17936245e-02   8.66400000e+01]	[  4.21077865e-04   2.16573313e+00]	[ -1.46958460e-02   8.20000000e+01]	[ -1.16406841e-02   9.70000000e+01]
    166	79    	[ -1.18023784e-02   8.64900000e+01]	[  3.83989872e-04   1.59684063e+00]	[ -1.43397074e-02   8.40000000e+01]	[ -1.16406841e-02   9.60000000e+01]
    167	75    	[ -1.19098548e-02   8.67000000e+01]	[  9.85672305e-04   2.17025344e+00]	[ -2.09145084e-02   8.20000000e+01]	[ -1.16406841e-02   9.80000000e+01]
    168	68    	[ -1.18781921e-02   8.66300000e+01]	[  4.50783055e-04   1.99326366e+00]	[ -1.35433147e-02   8.00000000e+01]	[ -1.16406841e-02   9.30000000e+01]
    169	82    	[ -1.18314599e-02   8.67000000e+01]	[  3.96936707e-04   1.90525589e+00]	[ -1.39016762e-02   8.10000000e+01]	[ -1.16406841e-02   9.50000000e+01]
    170	70    	[ -1.18752883e-02   8.64200000e+01]	[  7.24105494e-04   2.06484866e+00]	[ -1.76090232e-02   8.10000000e+01]	[ -1.16406841e-02   9.70000000e+01]
    171	70    	[ -1.18663980e-02   8.61800000e+01]	[  9.83697139e-04   1.00379281e+00]	[ -1.89113029e-02   8.20000000e+01]	[ -1.16406841e-02   9.20000000e+01]
    172	71    	[ -1.17388958e-02   8.63200000e+01]	[  2.86475503e-04   1.69044373e+00]	[ -1.32490094e-02   8.30000000e+01]	[ -1.16406841e-02   9.80000000e+01]
    173	85    	[ -1.18036084e-02   8.64700000e+01]	[  3.23959152e-04   1.80252601e+00]	[ -1.31528895e-02   8.30000000e+01]	[ -1.16260062e-02   9.20000000e+01]
    174	78    	[ -1.19111687e-02   8.68100000e+01]	[  6.75892005e-04   1.99346432e+00]	[ -1.72731905e-02   8.30000000e+01]	[ -1.16285237e-02   9.30000000e+01]
    175	78    	[ -1.19161876e-02   8.66300000e+01]	[  7.73355195e-04   1.82019230e+00]	[ -1.73388478e-02   8.30000000e+01]	[ -1.16285237e-02   9.50000000e+01]
    176	82    	[ -1.18591559e-02   8.67800000e+01]	[  6.97418337e-04   2.32198191e+00]	[ -1.75650484e-02   8.30000000e+01]	[ -1.16261265e-02   9.80000000e+01]
    177	71    	[ -1.19604952e-02   8.67700000e+01]	[  9.51901014e-04   2.04379549e+00]	[ -1.71093807e-02   8.30000000e+01]	[ -1.16261265e-02   9.70000000e+01]
    178	90    	[ -1.18000606e-02   8.65500000e+01]	[  4.43371884e-04   1.97167442e+00]	[ -1.47216097e-02   8.30000000e+01]	[ -1.16261265e-02   9.40000000e+01]
    179	73    	[ -1.18479697e-02   8.62300000e+01]	[  5.38234582e-04   1.68436932e+00]	[ -1.56588945e-02   8.10000000e+01]	[ -1.16261265e-02   9.20000000e+01]
    180	77    	[ -1.18230761e-02   8.64100000e+01]	[  6.34672128e-04   1.74982856e+00]	[ -1.73410153e-02   8.30000000e+01]	[ -1.16261265e-02   9.50000000e+01]
    181	73    	[ -1.18673480e-02   8.66100000e+01]	[  7.01215012e-04   1.77141187e+00]	[ -1.57591298e-02   8.30000000e+01]	[ -1.16261265e-02   9.60000000e+01]
    182	79    	[ -1.17445291e-02   8.66900000e+01]	[  2.96925313e-04   2.15728997e+00]	[ -1.34337018e-02   8.40000000e+01]	[ -1.16261265e-02   1.00000000e+02]
    183	81    	[ -1.19225487e-02   8.64600000e+01]	[  9.98235117e-04   1.79677489e+00]	[ -2.03104868e-02   8.10000000e+01]	[ -1.16261265e-02   9.40000000e+01]
    184	79    	[ -1.18039680e-02   8.64200000e+01]	[  5.19134647e-04   1.41548578e+00]	[ -1.58621245e-02   8.40000000e+01]	[ -1.16261265e-02   9.50000000e+01]
    185	71    	[ -1.17187907e-02   8.61500000e+01]	[  2.69493595e-04   1.14345966e+00]	[ -1.28134653e-02   8.30000000e+01]	[ -1.16261265e-02   9.20000000e+01]
    186	80    	[ -1.17289845e-02   8.62500000e+01]	[  3.05487471e-04   1.49248116e+00]	[ -1.35346863e-02   8.10000000e+01]	[ -1.16261265e-02   9.50000000e+01]
    187	82    	[ -1.18342069e-02   8.64900000e+01]	[  4.72153659e-04   1.86276676e+00]	[ -1.45346085e-02   8.00000000e+01]	[ -1.16261265e-02   9.30000000e+01]
    188	79    	[ -1.17695275e-02   8.64600000e+01]	[  3.25826468e-04   1.55190206e+00]	[ -1.32239171e-02   8.40000000e+01]	[ -1.16261265e-02   9.50000000e+01]
    189	86    	[ -1.18846107e-02   8.66400000e+01]	[  5.40088455e-04   1.78056171e+00]	[ -1.43169858e-02   8.10000000e+01]	[ -1.16261265e-02   9.30000000e+01]
    190	73    	[ -1.17543824e-02   8.64500000e+01]	[  3.06821921e-04   1.71099386e+00]	[ -1.31426136e-02   8.30000000e+01]	[ -1.16261265e-02   9.60000000e+01]
    191	85    	[ -1.18957833e-02   8.68300000e+01]	[  7.65511745e-04   2.16820202e+00]	[ -1.75909981e-02   8.10000000e+01]	[ -1.16261265e-02   9.70000000e+01]
    192	80    	[ -1.17550215e-02   8.66400000e+01]	[  3.12791414e-04   2.33032187e+00]	[ -1.31545495e-02   7.90000000e+01]	[ -1.16261265e-02   1.00000000e+02]
    193	75    	[ -1.18142526e-02   8.64700000e+01]	[  6.97784500e-04   1.59659012e+00]	[ -1.67477452e-02   8.20000000e+01]	[ -1.16261265e-02   9.50000000e+01]
    194	71    	[ -1.17861285e-02   8.66200000e+01]	[  5.31416608e-04   1.84271539e+00]	[ -1.63219926e-02   8.30000000e+01]	[ -1.16261265e-02   9.50000000e+01]
    195	80    	[ -1.17918866e-02   8.63600000e+01]	[  3.68759499e-04   1.44582157e+00]	[ -1.32743516e-02   8.30000000e+01]	[ -1.16261265e-02   9.30000000e+01]
    196	78    	[ -1.17468379e-02   8.64000000e+01]	[  3.24152648e-04   1.48996644e+00]	[ -1.32435506e-02   8.40000000e+01]	[ -1.16261265e-02   9.50000000e+01]
    197	69    	[ -1.18584477e-02   8.63300000e+01]	[  7.32418932e-04   1.53658713e+00]	[ -1.69759038e-02   8.20000000e+01]	[ -1.16261265e-02   9.40000000e+01]
    198	85    	[ -1.19514713e-02   8.63200000e+01]	[  9.37191917e-04   1.42744527e+00]	[ -1.96251856e-02   8.30000000e+01]	[ -1.16261265e-02   9.10000000e+01]
    199	72    	[ -1.18142254e-02   8.63200000e+01]	[  5.45902319e-04   1.76567268e+00]	[ -1.60603128e-02   8.00000000e+01]	[ -1.16261265e-02   9.40000000e+01]
    200	80    	[ -1.17815962e-02   8.64600000e+01]	[  5.03635351e-04   1.57111425e+00]	[ -1.62046471e-02   8.20000000e+01]	[ -1.16261265e-02   9.30000000e+01]
    201	81    	[ -1.18262026e-02   8.63400000e+01]	[  5.20136599e-04   1.56345771e+00]	[ -1.59557538e-02   7.90000000e+01]	[ -1.16261265e-02   9.20000000e+01]
    202	80    	[ -1.17646197e-02   8.63900000e+01]	[  4.04407098e-04   1.61180024e+00]	[ -1.44980723e-02   8.10000000e+01]	[ -1.16261265e-02   9.30000000e+01]
    203	76    	[ -1.18036578e-02   8.66500000e+01]	[  6.40397737e-04   1.88878268e+00]	[ -1.61448742e-02   8.40000000e+01]	[ -1.16261265e-02   9.70000000e+01]
    204	78    	[ -1.17915306e-02   8.65800000e+01]	[  5.41462136e-04   1.81758081e+00]	[ -1.56295047e-02   8.30000000e+01]	[ -1.16261265e-02   9.50000000e+01]
    205	65    	[ -1.17922889e-02   8.64300000e+01]	[  4.38822709e-04   1.64471882e+00]	[ -1.43147604e-02   8.30000000e+01]	[ -1.16261265e-02   9.80000000e+01]
    206	77    	[ -1.18551088e-02   8.65300000e+01]	[  6.92933567e-04   1.67005988e+00]	[ -1.60752524e-02   8.40000000e+01]	[ -1.16261265e-02   9.40000000e+01]
    207	72    	[ -1.17969868e-02   8.62400000e+01]	[  5.66359762e-04   1.49746452e+00]	[ -1.62615280e-02   8.00000000e+01]	[ -1.16261265e-02   9.30000000e+01]
    208	80    	[ -1.17851465e-02   8.63900000e+01]	[  3.90032157e-04   1.83790642e+00]	[ -1.38777343e-02   8.20000000e+01]	[ -1.16261265e-02   9.60000000e+01]
    209	80    	[ -1.18059394e-02   8.63100000e+01]	[  6.30032993e-04   1.71286310e+00]	[ -1.72441424e-02   7.90000000e+01]	[ -1.16261265e-02   9.40000000e+01]
    210	74    	[ -1.17210349e-02   8.65300000e+01]	[  2.54894848e-04   1.41035457e+00]	[ -1.30318225e-02   8.60000000e+01]	[ -1.16261265e-02   9.30000000e+01]
    211	80    	[ -1.17697878e-02   8.67800000e+01]	[  3.42323172e-04   2.10988151e+00]	[ -1.37684030e-02   8.40000000e+01]	[ -1.16261265e-02   9.50000000e+01]
    212	77    	[ -1.18536586e-02   8.65500000e+01]	[  6.13659155e-04   2.00686322e+00]	[ -1.63180541e-02   8.20000000e+01]	[ -1.16261265e-02   9.80000000e+01]
    213	72    	[ -1.17574518e-02   8.63400000e+01]	[  5.58453947e-04   1.64450600e+00]	[ -1.68502155e-02   8.20000000e+01]	[ -1.16261265e-02   9.40000000e+01]
    214	77    	[ -1.17535227e-02   8.66800000e+01]	[  3.24154550e-04   1.85407659e+00]	[ -1.35094897e-02   8.50000000e+01]	[ -1.16261265e-02   9.60000000e+01]
    215	87    	[ -1.18277548e-02   8.65300000e+01]	[  5.73918864e-04   1.55855702e+00]	[ -1.58820476e-02   8.30000000e+01]	[ -1.16261265e-02   9.20000000e+01]
    216	68    	[ -1.17197476e-02   8.64300000e+01]	[  2.94447045e-04   1.28261452e+00]	[ -1.34114273e-02   8.50000000e+01]	[ -1.16261265e-02   9.40000000e+01]
    217	75    	[ -1.18212684e-02   8.65500000e+01]	[  4.57627983e-04   1.91507180e+00]	[ -1.39088021e-02   8.00000000e+01]	[ -1.16261265e-02   9.30000000e+01]
    218	83    	[ -1.18859249e-02   8.65100000e+01]	[  7.82156186e-04   1.72913273e+00]	[ -1.60689037e-02   8.30000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    219	76    	[ -1.17358816e-02   8.64600000e+01]	[  2.89601455e-04   1.52590956e+00]	[ -1.29547996e-02   8.40000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    220	71    	[ -1.19406449e-02   8.68600000e+01]	[  7.00620179e-04   1.91321719e+00]	[ -1.57966087e-02   8.30000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    221	74    	[ -1.17745182e-02   8.65100000e+01]	[  3.47621049e-04   1.65224090e+00]	[ -1.36410333e-02   8.40000000e+01]	[ -1.16253005e-02   9.30000000e+01]
    222	75    	[ -1.17488679e-02   8.69400000e+01]	[  2.92586063e-04   2.09198470e+00]	[ -1.31962622e-02   8.50000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    223	74    	[ -1.18666051e-02   8.68200000e+01]	[  7.36404669e-04   1.41689802e+00]	[ -1.66039978e-02   8.40000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    224	83    	[ -1.18370416e-02   8.71900000e+01]	[  6.53946651e-04   1.78154427e+00]	[ -1.58593043e-02   8.30000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    225	78    	[ -1.18551726e-02   8.72500000e+01]	[  7.21912521e-04   1.49916644e+00]	[ -1.64767664e-02   8.20000000e+01]	[ -1.16253005e-02   9.30000000e+01]
    226	78    	[ -1.18573772e-02   8.74600000e+01]	[  5.87430869e-04   1.74022987e+00]	[ -1.60303345e-02   8.30000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    227	74    	[ -1.17917742e-02   8.75900000e+01]	[  6.51261223e-04   1.88199362e+00]	[ -1.77485818e-02   8.40000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    228	81    	[ -1.18459853e-02   8.74600000e+01]	[  7.44512525e-04   1.55833244e+00]	[ -1.80898734e-02   8.20000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    229	67    	[ -1.17958132e-02   8.74400000e+01]	[  4.52346401e-04   1.77380946e+00]	[ -1.45686676e-02   8.20000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    230	78    	[ -1.18447088e-02   8.75900000e+01]	[  6.76115562e-04   1.70935660e+00]	[ -1.63686641e-02   8.60000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    231	69    	[ -1.18457075e-02   8.74100000e+01]	[  5.60505977e-04   1.64981817e+00]	[ -1.57436924e-02   8.20000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    232	81    	[ -1.17559164e-02   8.72800000e+01]	[  3.48251422e-04   1.39341308e+00]	[ -1.33836269e-02   8.50000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    233	75    	[ -1.19239316e-02   8.73400000e+01]	[  9.60424275e-04   1.59511755e+00]	[ -1.83486239e-02   8.00000000e+01]	[ -1.16253005e-02   9.30000000e+01]
    234	81    	[ -1.18606268e-02   8.77100000e+01]	[  5.97957524e-04   2.18309413e+00]	[ -1.61249812e-02   8.00000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    235	73    	[ -1.18161441e-02   8.76500000e+01]	[  4.80472275e-04   1.89934199e+00]	[ -1.39308521e-02   8.40000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    236	72    	[ -1.17766648e-02   8.75600000e+01]	[  3.34119393e-04   1.59574434e+00]	[ -1.32257062e-02   8.40000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    237	68    	[ -1.17939445e-02   8.74300000e+01]	[  4.32145448e-04   1.84529130e+00]	[ -1.40634108e-02   7.90000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    238	80    	[ -1.17634187e-02   8.73900000e+01]	[  4.74125304e-04   1.73144448e+00]	[ -1.51485528e-02   8.20000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    239	73    	[ -1.17997493e-02   8.74600000e+01]	[  4.99795049e-04   1.72869893e+00]	[ -1.57716977e-02   8.30000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    240	72    	[ -1.19323653e-02   8.73600000e+01]	[  1.12533654e-03   1.49345238e+00]	[ -2.03572380e-02   8.30000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    241	76    	[ -1.18428092e-02   8.74000000e+01]	[  6.50500944e-04   1.59373775e+00]	[ -1.61497674e-02   8.20000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    242	77    	[ -1.18563565e-02   8.73700000e+01]	[  5.91125460e-04   1.67722986e+00]	[ -1.62494984e-02   8.10000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    243	79    	[ -1.18571643e-02   8.72500000e+01]	[  7.75918772e-04   1.25199840e+00]	[ -1.76132081e-02   8.20000000e+01]	[ -1.16253005e-02   9.30000000e+01]
    244	78    	[ -1.17865781e-02   8.76000000e+01]	[  3.14944926e-04   1.75499288e+00]	[ -1.29798141e-02   8.20000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    245	75    	[ -1.17661994e-02   8.79200000e+01]	[  3.72184983e-04   2.15257985e+00]	[ -1.39065960e-02   8.60000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    246	73    	[ -1.18602319e-02   8.74900000e+01]	[  7.45299136e-04   2.10947861e+00]	[ -1.70386842e-02   8.10000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    247	81    	[ -1.18940853e-02   8.74800000e+01]	[  9.78907298e-04   1.55871742e+00]	[ -1.96314033e-02   8.40000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    248	85    	[ -1.18384231e-02   8.75900000e+01]	[  5.61470733e-04   1.85523583e+00]	[ -1.56218550e-02   8.40000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    249	65    	[ -1.17878563e-02   8.74100000e+01]	[  3.95440060e-04   1.64981817e+00]	[ -1.39420688e-02   8.40000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    250	76    	[ -1.18395564e-02   8.73800000e+01]	[  6.04719098e-04   1.89092570e+00]	[ -1.57972883e-02   8.10000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    251	77    	[ -1.19879876e-02   8.71100000e+01]	[  8.53302135e-04   1.77141187e+00]	[ -1.77720476e-02   8.00000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    252	72    	[ -1.19057438e-02   8.71200000e+01]	[  8.45196712e-04   1.89356806e+00]	[ -1.82038788e-02   8.20000000e+01]	[ -1.16253005e-02   9.80000000e+01]
    253	80    	[ -1.17694215e-02   8.73800000e+01]	[  5.03495419e-04   1.54129815e+00]	[ -1.56450194e-02   8.40000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    254	69    	[ -1.17710166e-02   8.70500000e+01]	[  5.60399024e-04   1.01365675e+00]	[ -1.67565202e-02   8.10000000e+01]	[ -1.16253005e-02   9.10000000e+01]
    255	73    	[ -1.17906838e-02   8.72100000e+01]	[  5.77515298e-04   1.76235638e+00]	[ -1.63433811e-02   7.90000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    256	85    	[ -1.17612338e-02   8.75100000e+01]	[  3.49119328e-04   1.79719225e+00]	[ -1.33614584e-02   8.20000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    257	73    	[ -1.17881315e-02   8.71700000e+01]	[  6.47791719e-04   9.90504922e-01]	[ -1.59421908e-02   8.40000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    258	83    	[ -1.16934289e-02   8.76900000e+01]	[  1.80607549e-04   2.06734129e+00]	[ -1.24858176e-02   8.60000000e+01]	[ -1.16253005e-02   9.90000000e+01]
    259	77    	[ -1.17755049e-02   8.75600000e+01]	[  3.31012998e-04   1.77943811e+00]	[ -1.34184505e-02   8.50000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    260	71    	[ -1.18262876e-02   8.74900000e+01]	[  4.63142993e-04   1.71752729e+00]	[ -1.43871607e-02   8.30000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    261	81    	[ -1.18254699e-02   8.75100000e+01]	[  5.57954601e-04   1.87347271e+00]	[ -1.60544567e-02   8.30000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    262	71    	[ -1.18877690e-02   8.75300000e+01]	[  6.62351901e-04   1.69973527e+00]	[ -1.63087269e-02   8.20000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    263	73    	[ -1.18279964e-02   8.75400000e+01]	[  5.33514449e-04   1.51934196e+00]	[ -1.47338956e-02   8.40000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    264	75    	[ -1.18016060e-02   8.73900000e+01]	[  6.28094443e-04   1.81049717e+00]	[ -1.69335747e-02   8.10000000e+01]	[ -1.16253005e-02   9.80000000e+01]
    265	72    	[ -1.18500573e-02   8.73700000e+01]	[  5.82686262e-04   1.65924682e+00]	[ -1.58490398e-02   8.30000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    266	82    	[ -1.18779923e-02   8.74600000e+01]	[  7.38677985e-04   1.70540318e+00]	[ -1.56840997e-02   8.30000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    267	77    	[ -1.18558661e-02   8.74400000e+01]	[  7.27953128e-04   1.81284307e+00]	[ -1.68967571e-02   8.10000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    268	76    	[ -1.18227960e-02   8.75900000e+01]	[  5.35992444e-04   1.94471078e+00]	[ -1.56034938e-02   8.30000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    269	84    	[ -1.19295602e-02   8.77600000e+01]	[  8.61040029e-04   1.99559515e+00]	[ -1.83606303e-02   8.20000000e+01]	[ -1.16253005e-02   9.80000000e+01]
    270	74    	[ -1.18259806e-02   8.77000000e+01]	[  4.41212380e-04   1.87882942e+00]	[ -1.42273497e-02   8.30000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    271	69    	[ -1.17774628e-02   8.74200000e+01]	[  5.52457870e-04   1.46410382e+00]	[ -1.54708018e-02   8.50000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    272	81    	[ -1.18049090e-02   8.73600000e+01]	[  5.27258463e-04   1.72927731e+00]	[ -1.44671520e-02   8.00000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    273	76    	[ -1.17643183e-02   8.74100000e+01]	[  3.10126666e-04   1.54334053e+00]	[ -1.30218362e-02   8.20000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    274	80    	[ -1.18117411e-02   8.75800000e+01]	[  4.89642348e-04   1.82307433e+00]	[ -1.55460831e-02   8.30000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    275	66    	[ -1.18805982e-02   8.75700000e+01]	[  8.19837160e-04   1.75074270e+00]	[ -1.80943931e-02   8.40000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    276	71    	[ -1.17488935e-02   8.72500000e+01]	[  4.20660372e-04   1.51904575e+00]	[ -1.48095332e-02   8.20000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    277	72    	[ -1.17774204e-02   8.72800000e+01]	[  3.85219462e-04   1.45657132e+00]	[ -1.40177658e-02   8.30000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    278	83    	[ -1.18147515e-02   8.75400000e+01]	[  5.08960685e-04   1.80233182e+00]	[ -1.56807477e-02   8.30000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    279	66    	[ -1.17793950e-02   8.76700000e+01]	[  3.27786574e-04   2.01521711e+00]	[ -1.29072179e-02   8.40000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    280	79    	[ -1.17660947e-02   8.72000000e+01]	[  4.69151640e-04   1.34907376e+00]	[ -1.56070946e-02   8.30000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    281	77    	[ -1.18168967e-02   8.76400000e+01]	[  5.65659944e-04   1.99759856e+00]	[ -1.61786314e-02   8.40000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    282	74    	[ -1.17429823e-02   8.72700000e+01]	[  3.55537880e-04   9.88483687e-01]	[ -1.37129278e-02   8.60000000e+01]	[ -1.16253005e-02   9.30000000e+01]
    283	77    	[ -1.17616401e-02   8.74700000e+01]	[  3.18834858e-04   1.84095084e+00]	[ -1.29476157e-02   8.00000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    284	73    	[ -1.19641734e-02   8.74700000e+01]	[  9.47195777e-04   1.72890139e+00]	[ -1.68935169e-02   8.40000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    285	68    	[ -1.17724445e-02   8.76800000e+01]	[  3.16146576e-04   2.55687309e+00]	[ -1.33305735e-02   8.10000000e+01]	[ -1.16253005e-02   9.80000000e+01]
    286	77    	[ -1.18119085e-02   8.74300000e+01]	[  5.51174534e-04   1.57006369e+00]	[ -1.57262912e-02   8.60000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    287	82    	[ -1.18423103e-02   8.77700000e+01]	[  4.78575348e-04   2.01422442e+00]	[ -1.44398970e-02   8.50000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    288	75    	[ -1.18031302e-02   8.75400000e+01]	[  5.54482031e-04   1.74022987e+00]	[ -1.59883373e-02   8.40000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    289	72    	[ -1.18234621e-02   8.72600000e+01]	[  6.99682408e-04   1.37564530e+00]	[ -1.71117302e-02   8.50000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    290	67    	[ -1.17912941e-02   8.72600000e+01]	[  5.62549183e-04   1.70070574e+00]	[ -1.62780810e-02   8.00000000e+01]	[ -1.16253005e-02   9.80000000e+01]
    291	79    	[ -1.17326180e-02   8.77600000e+01]	[  2.54809680e-04   2.10295031e+00]	[ -1.28083598e-02   8.30000000e+01]	[ -1.16253005e-02   1.01000000e+02]
    292	76    	[ -1.18142197e-02   8.74700000e+01]	[  5.55246178e-04   1.57133701e+00]	[ -1.58631116e-02   8.30000000e+01]	[ -1.16253005e-02   9.30000000e+01]
    293	88    	[ -1.18162890e-02   8.75000000e+01]	[  5.33313931e-04   1.47309199e+00]	[ -1.59198610e-02   8.60000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    294	81    	[ -1.18462753e-02   8.75000000e+01]	[  7.24582048e-04   1.68226038e+00]	[ -1.67022911e-02   8.30000000e+01]	[ -1.16253005e-02   9.50000000e+01]
    295	77    	[ -1.18814544e-02   8.76900000e+01]	[  8.05049788e-04   1.78714857e+00]	[ -1.63742205e-02   8.50000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    296	69    	[ -1.17640275e-02   8.77000000e+01]	[  3.10248032e-04   1.87882942e+00]	[ -1.32375121e-02   8.50000000e+01]	[ -1.16253005e-02   9.70000000e+01]
    297	68    	[ -1.17838084e-02   8.71500000e+01]	[  4.14977706e-04   1.09886305e+00]	[ -1.35235937e-02   8.30000000e+01]	[ -1.16253005e-02   9.10000000e+01]
    298	73    	[ -1.18681482e-02   8.77500000e+01]	[  6.93099915e-04   1.86212244e+00]	[ -1.60154883e-02   8.50000000e+01]	[ -1.16253005e-02   9.60000000e+01]
    299	72    	[ -1.18419304e-02   8.75200000e+01]	[  7.17117908e-04   1.75772580e+00]	[ -1.81720853e-02   8.30000000e+01]	[ -1.16253005e-02   9.40000000e+01]
    300	78    	[ -1.18146701e-02   8.73700000e+01]	[  7.60569273e-04   1.59157155e+00]	[ -1.75347452e-02   8.20000000e+01]	[ -1.16253005e-02   9.70000000e+01]


##### 又一次耗时两天。。。


```python
print compute_score(ENet, train_data[train_data.columns[supports_[0]]].values, np.log(SalePrice.values))
print compute_score(GBoost, train_data[train_data.columns[supports_[1]]].values, np.log(SalePrice.values))
print compute_score(lasso, train_data[train_data.columns[supports_[2]]].values, np.log(SalePrice.values))
print compute_score(model_xgb, train_data[train_data.columns[supports_[3]]].values, np.log(SalePrice.values))
print compute_score(KRR, train_data[train_data.columns[supports_[4]]].values, np.log(SalePrice.values))
```

    (0.11567711551720301, 0.0084931598160457936)
    (0.11534675639270017, 0.0081688179658207221)
    (0.11588976622128115, 0.0080453441841918053)
    (0.11653378136631638, 0.0082873550442852436)
    (0.11027129268374732, 0.0091623600281198134)



```python
averaged_models = DifferentFeatureAveragingModels(models = (ENet, GBoost, lasso, model_xgb, KRR), supports=supports_)
```


```python
compute_score(averaged_models, train_data.values, np.log(SalePrice.values))
```




    (0.084292592656284041, 0.0099428564968274836)



###### 发现结果从0.09-》0.08，看来集成多个模型还是有用的


```python
averaged_models.fit(train_data.values, np.log(SalePrice.values))
```




    DifferentFeatureAveragingModels(models=(Pipeline(steps=[('robustscaler', RobustScaler(copy=True, with_centering=True, with_scaling=True)), ('elasticnet', ElasticNet(alpha=0.0005, copy_X=True, fit_intercept=True, l1_ratio=0.9,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=3, selection='c...nelRidge(alpha=0.6, coef0=2.5, degree=2, gamma=None, kernel='polynomial',
          kernel_params=None)),
                    supports=[[0, 2, 7, 8, 9, 10, 11, 12, 13, 16, 18, 19, 23, 24, 25, 26, 28, 29, 30, 32, 35, 37, 38, 40, 41, 42, 43, 44, 47, 48, 51, 56, 58, 60, 61, 62, 67, 68, 69, 73, 75, 78, 79, 87, 88, 93, 94, 96, 98, 99, 104, 105, 110, 114, 115, 116, 117, 118, 122, 129, 130, 131, 134, 137, 138, 141, 143, 148, 150,...154, 162, 164, 169, 171, 174, 179, 182, 183, 186, 191, 198, 200, 201, 204, 206, 208, 209, 212, 213]])




```python
res = np.exp(averaged_models.predict(test_data.values))
```


```python
pd.DataFrame({'Id': Id, 'SalePrice': res}).to_csv('2017-8-21-9-06.csv', index=False)
```

##### 结论
- 这次的结果上传上去有0.12，top 20%
- 虽然还是非常水的成绩，不过最近还有别的事情要忙，所以就先告一段落
- 这次使用了GA来进行特征选择，效果还是很不错的，日后如果有机会接触一下进化算法的特征选择和集成学习也是蛮好玩的
- 下次有机会可以用神经网络来进行预测，虽然这次的数据集感觉上来讲还是满规范的。
- 第二次kaggle之旅，收获颇丰，而且也挺好玩哈哈哈。


```python

```
