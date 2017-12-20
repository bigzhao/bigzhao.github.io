---
layout:     post
title:      "记首次kaggle夺银经历"
date:       2017-12-18 19:19:00
author:     "Bigzhao"
header-img: "img/post-bg-02.jpg"
---

## Introduction
Kaggle 是目前最大的 Data Scientist 聚集地。很多公司会拿出自家的数据并提供奖金，在 Kaggle 上组织数据竞赛。我最近完成了第一次比赛,，在 5,169 个参赛队伍中排名第 131 位（top3%,首次夺银哈哈哈）。因为是第一次参赛，所以对这个成绩我已经很满意了。

接下来简单地介绍这个比赛，我参加的比赛叫做Porto Seguro’s Safe Driver Prediction，主要是根据历史数据建立一个模型，预测驾驶员将在明年发起汽车保险索赔的可能性。比赛的评估标准是gini系数。

## Data Exploration
公开kernel上有比较详细的数据可视化教程[[点我]](https://www.kaggle.com/bertcarremans/data-preparation-exploration)

由于该比赛的数据特征的真实含义被隐藏，因此无法在特征工程上下太大功夫。

## 数据预处理
基于oliver的kernel,根据相关度去掉ps_calc一系列特征
```py
# from olivier
train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
	"ps_reg_03",  #            : 1408.42 / shadow  511.15
	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
	"ps_ind_03",  #            : 1219.47 / shadow  230.55
	"ps_ind_15",  #            :  922.18 / shadow  242.00
	"ps_reg_02",  #            :  920.65 / shadow  267.50
	"ps_car_14",  #            :  798.48 / shadow  549.58
	"ps_car_12",  #            :  731.93 / shadow  293.62
	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
	"ps_car_03_cat",  #        :  611.73 / shadow   50.67
	"ps_reg_01",  #            :  598.60 / shadow  178.57
	"ps_car_15",  #            :  593.35 / shadow  226.43
	"ps_ind_01",  #            :  547.32 / shadow  154.58
	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
	"ps_car_11",  #            :  173.28 / shadow   76.45
	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
	"ps_calc_09",  #           :  169.13 / shadow  129.72
	"ps_calc_05",  #           :  148.83 / shadow  120.68
	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
	"ps_ind_14",  #            :   37.37 / shadow   16.65
]
# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),  
    ('ps_reg_01', 'ps_car_04_cat'),
]
# Process data
id_test = test_df['id'].values
id_train = train_df['id'].values
y = train_df['target']

start = time.time()
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60), end='')
    print('\r' * 75, end='')
    train_df[name1] = train_df[f1].apply(lambda x: str(x)) + "_" + train_df[f2].apply(lambda x: str(x))
    test_df[name1] = test_df[f1].apply(lambda x: str(x)) + "_" + test_df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
    train_df[name1] = lbl.transform(list(train_df[name1].values))
    test_df[name1] = lbl.transform(list(test_df[name1].values))

    train_features.append(name1)

X = train_df[train_features]
test_df = test_df[train_features]

f_cats = [f for f in X.columns if "_cat" in f]
```
## 模型选择

在这次比赛中，尝试了DNN,XGBOOST,LIGHTGBM,CATBOOST,LIBFFM,模型聚合尝试了stacking，voting，blending。。结果发现一开始voting的结果是最好的，但是后面加模型的时候stacking就比voting好了。最后的提交方案是DNN+stacking+voting+根据public成绩比较靠前的几次结果的平均值

其中DNN是用了别人的模型，然后稍微修改了一下。模型代码如下：
```py

def build_embedding_network():

    models = []

    model_ps_ind_02_cat = Sequential()
    model_ps_ind_02_cat.add(Embedding(5, 3, input_length=1))
    model_ps_ind_02_cat.add(Reshape(target_shape=(3,)))
    models.append(model_ps_ind_02_cat)

    model_ps_ind_04_cat = Sequential()
    model_ps_ind_04_cat.add(Embedding(3, 2, input_length=1))
    model_ps_ind_04_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_ind_04_cat)

    model_ps_ind_05_cat = Sequential()
    model_ps_ind_05_cat.add(Embedding(8, 5, input_length=1))
    model_ps_ind_05_cat.add(Reshape(target_shape=(5,)))
    models.append(model_ps_ind_05_cat)

    model_ps_car_01_cat = Sequential()
    model_ps_car_01_cat.add(Embedding(13, 7, input_length=1))
    model_ps_car_01_cat.add(Reshape(target_shape=(7,)))
    models.append(model_ps_car_01_cat)

    model_ps_car_02_cat = Sequential()
    model_ps_car_02_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_02_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_02_cat)

    model_ps_car_03_cat = Sequential()
    model_ps_car_03_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_03_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_03_cat)

    model_ps_car_04_cat = Sequential()
    model_ps_car_04_cat.add(Embedding(10, 5, input_length=1))
    model_ps_car_04_cat.add(Reshape(target_shape=(5,)))
    models.append(model_ps_car_04_cat)

    model_ps_car_05_cat = Sequential()
    model_ps_car_05_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_05_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_05_cat)

    model_ps_car_06_cat = Sequential()
    model_ps_car_06_cat.add(Embedding(18, 8, input_length=1))
    model_ps_car_06_cat.add(Reshape(target_shape=(8,)))
    models.append(model_ps_car_06_cat)

    model_ps_car_07_cat = Sequential()
    model_ps_car_07_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_07_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_07_cat)

    model_ps_car_09_cat = Sequential()
    model_ps_car_09_cat.add(Embedding(6, 3, input_length=1))
    model_ps_car_09_cat.add(Reshape(target_shape=(3,)))
    models.append(model_ps_car_09_cat)

    model_ps_car_10_cat = Sequential()
    model_ps_car_10_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_10_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_10_cat)

    model_ps_car_11_cat = Sequential()
    model_ps_car_11_cat.add(Embedding(104, 10, input_length=1))
    model_ps_car_11_cat.add(Reshape(target_shape=(10,)))
    models.append(model_ps_car_11_cat)

    model_rest = Sequential()
    model_rest.add(Dense(16, input_dim=24))
    models.append(model_rest)

    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(.35))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
```

## 调参

使用random search方法调参，这样的好处是能给我带来多组不错的参数。

下面是随机调xgboost的例子
```py
params_set = []
ginis = []
OPTIMIZE_ROUNDS = True
for i in range(0, 10):
    param_dist = {
            'max_depth': 4,     
            'objective': 'binary:logistic',
            'learning_rate': choice([0.2, 0.1, 0.07, 0.06, 0.08, 0.05, 0.03]),
            'subsample': choice([0.8, 0.75, 0.85, 0.7, 0.8]),
            'min_child_weight': choice([0.77, 0.7, 0.8, 0.9, 1]),
            'scale_pos_weight':choice([1, 1.3, 1.6, 1.7]),
            'gamma':10,
            'reg_alpha' : 8,
            'reg_lambda':1.3,
            'n_estimators': 1000
    }

    y_valid_pred = 0*y
    y_test_pred = 0
    xgb_model = xgb.XGBClassifier(**param_dist)

    for i, (train_index, test_index) in enumerate(kf.split(train_df)):
        # Create data for this fold
        y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
        X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
        X_test = test_df.copy()
        print( "\nFold ", i)

        # Enocode data
        for f in f_cats:
            X_train[f + "_avg"], X_valid[f + "_avg"], X_test[f + "_avg"] = target_encode(
                                                            trn_series=X_train[f],
                                                            val_series=X_valid[f],
                                                            tst_series=X_test[f],
                                                            target=y_train,
                                                            min_samples_leaf=200,
                                                            smoothing=10,
                                                            noise_level=0
                                                            )
        # Run model for this fold
        if OPTIMIZE_ROUNDS:
            eval_set=[(X_valid,y_valid)]
            fit_model = xgb_model.fit( X_train, y_train,
                                   eval_set=eval_set,
                                   eval_metric=gini_xgb,
                                   early_stopping_rounds=100,

                                   verbose = 100
                                 )
            print( "  Best N trees = ", fit_model.best_iteration )
            print( "  Best gini = ", fit_model.best_score )
        else:
            fit_model = xgb_model.fit( X_train, y_train )

        # Generate validation predictions for this fold
        pred = fit_model.predict_proba(X_valid)[:,1]
        print( "  Gini = ", eval_gini(y_valid, pred) )
        y_valid_pred.iloc[test_index] = pred

        # Accumulate test set predictions
        y_test_pred += fit_model.predict_proba(X_test)[:,1]

        del X_test, X_train, X_valid, y_train

    y_test_pred /= K  # Average test set predictions
    res =  eval_gini(y, y_valid_pred)
    print( "\nGini: {} params:{}:".format(res, ip)  )
    params_set.append(param_dist.copy())
    ginis.append(res)
```

## Submit
因为评估标准是gini，所以最后采取rank的方法去做加权平均。
```py
indir = 'input\my_submit/'
infiles = [
'NN_EntityEmbed_10fold-sub.csv', 'voting_res_mean.csv', 'stack_test_8folds.csv', 'rank.csv','stack_002_test.csv'
]
for i, f in enumerate(infiles):
    subf = pd.read_csv(indir + infiles[i])
    if not i:
        sub = subf
    else:
        sub = pd.merge(sub, subf, on='id', suffixes=['',str(i)])
# oof.rename(columns={'target':'target0'}, inplace=True)
# oof.head()
sub['target'] = (sub.drop('id', axis=1).rank() / sub.shape[0]).mean(axis=1)
sub[['id', 'target']].to_csv('my_submission_mix.csv', index=False)
```

## 总结

- 计算资源很重要，没有GPU很无奈。。。
- 多关注比赛论坛，开工kernel有很多很好的idea
- 怎样权衡public成绩和自己的cv成绩。。。这场比赛很多人overfitting可能就是因为太过于看重public排名。。
- 中间尝试blending的时候成绩非常好，但是public排名超级烂，可能的原因是我blending没有使用同一个CV导致数据泄露，这个还得研究研究。。。
- 看到排名前三运用神经网络非常厉害，在我这里神经网络得成绩怎么调都比不过boost树，还得继续学习学习。。
- stacking/voting的时候少部分是不同的模型，大部分是相同模型+不同参数，不知道这样会不会相关度太高而导致stacking的作用没有很好地发挥。
- 尝试过两层stacking，后来发现实在是太慢了跑不下去了。。。
