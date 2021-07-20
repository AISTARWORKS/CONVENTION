# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ## Devices Life Prediction Version 0.2
# #### Developed by A.Okada, T.Shirakami, K.Kuramitsu, K.Iino and N.Yamazaki
# #### Copyright© A.Okada, T.Shirakami, K.Kuramitsu, K.Iino and N.Yamazaki, 2021
# #### Start from June 8, 2021
# 
# #### https://github.com/AISTARWORKS/CONVENTION.git
# 
# git push --set-upstream origin master
# git push origin master
# %% [markdown]
# ### Sample Generator

# %%
# Library import
import pandas as pd # 表=DataFrameに関する
import numpy as np # 行列に関する
import random # ramdom値生成
import math # 数学演算、logとかsin, cosとか
import matplotlib.pyplot as plt # グラフに関する
import matplotlib as mpl # 同じくグラフ関係、要らないかも。
import seaborn as sns

# DataFrameの値の小数点以下桁数はここで調整。
pd.options.display.precision = 2

Sampling = 1000
NumSample = 100
ArrheniusA = np.e
ArrheniusB = 1000
ArrheniusC = 0.035

# 複数FANタイプに対応するため関数化。
def sample_generator(RpmSpec,
                     PowerSpec,
                     TempSpec,
                     DiameterSpec,
                     ThicknessSpec,
                     QSpec,
                     PSpec,
                     LifeSpec,
                     Sampling,
                     NumSample):

    data_list = [[]] # sampling毎のlistなので2次元list
#     cum_rpm = 0
#     cum_temp = 0
#     cum_power = 0

    for sample_id in range(NumSample):

        rpm = RpmSpec # 初期値。最初から0だと困る。
        cum_rpm = 0
        cum_temp = 0
        cum_power = 0
        cumurated_life_impact_factor = 0

        if np.random.random() < 0.1: # ランダムに10%は寿命の短い不良品が混入  ==>rpmが下がりpowerが増える
            defect = 1
        else:
            defect = 0

        for time in range(Sampling, LifeSpec*2, Sampling):

            temp = 25 + random.uniform(-5,5)

            if rpm <= 0: # 前のforループで死んでたら後は永久に死。
                power = 0
                death = 1
            else:

                if defect == 1: # 不良品の場合
                    # rpm; 40degCでフル回転, 時間とともに低下、 +/-5% 誤差考慮でランダム
                    rpm = (-1 * ((time + Sampling)/8000) ** 6 + RpmSpec * temp / TempSpec) * (1-random.uniform(-0.05,0.05))
                    if rpm < 0.1*RpmSpec:
                        rpm = 0
                        power = 0
                        death = 1
                        remaining_life = 0
                    else:   
                        #power: rpmの低下により増加する成分と、温度に追従する成分をもつ。+/-5%誤差考慮でランダム 
                        power = (0.5 * (4000/rpm) ** 1.2 + PowerSpec * (temp / TempSpec)) * (1 - random.uniform(-0.05,0.05))
                        death = 0
                        # remaining_life
                        k = ArrheniusA ** (ArrheniusB/(273 + temp)) * ArrheniusC
                        remaining_life = k * ((rpm*temp/TempSpec - 0.1*cum_rpm/time)**(1/6)) * 8000 * (1 - random.uniform(-0.05,0.05)) - time
                        if remaining_life < 0:
                            remaining_life = 0
                else: # 良品の場合
                    rpm = (-1 * ((time + Sampling)/8000) ** 4 + RpmSpec * temp / TempSpec) * (1-random.uniform(-0.05,0.05))
                    if rpm < 100:
                        rpm = 0
                        power = 0
                        death = 1
                        remaining_life = 0
                    else: 
                        power = (0.5 * (4000/rpm) ** 1.1 + PowerSpec * (temp / TempSpec)) * (1 - random.uniform(-0.05,0.05))
                        death = 0
                        k = ArrheniusA ** (ArrheniusB/(273 + temp)) * ArrheniusC
                        remaining_life = k * ((rpm*temp/TempSpec - 0.1*cum_rpm/time)**(1/4)) * 8000 * (1 - random.uniform(-0.05,0.05)) -time
                        if remaining_life < 0:
                            remaining_life = 0

            # 累積。cum = cumurated = 累積
            cum_rpm += rpm
            cum_temp += temp
            cum_power += power

            # FANの寿命に与えるファクタとして、累積rpmの逆数, 累積temp, 累積powerの積の対数
            # cumurated_life_impact_factorは、0を中心に+/-1以内で推移。大きいと寿命に与える影響大。

            cumurated_life_impact_factor = math.log(10, ((1/cum_rpm) ** 0.5) * cum_temp * cum_power)

            data_list.append([sample_id,
                            defect,
                            time, 
                            rpm, 
                            temp, 
                            power, 
                            cum_rpm, 
                            cum_temp, 
                            cum_power, 
                            RpmSpec, 
                            PowerSpec, 
                            DiameterSpec, 
                            ThicknessSpec, 
                            QSpec, 
                            PSpec, 
                            LifeSpec, 
                            cumurated_life_impact_factor, 
                            death,
                            k,
                            remaining_life])
    return data_list

fan40 = sample_generator(RpmSpec = 25000,
                        PowerSpec = 20.16,
                        TempSpec = 40,
                        DiameterSpec = 40,
                        ThicknessSpec = 28,
                        QSpec = 0.83, # m^3/min
                        PSpec = 1100, # Pa
                        LifeSpec = 40000,
                        Sampling = Sampling,
                        NumSample = NumSample)

fan40cr = sample_generator(RpmSpec = 22000,
                        PowerSpec = 19.2,
                        TempSpec = 40,
                        DiameterSpec = 40,
                        ThicknessSpec = 56,
                        QSpec = 0.9, # m^3/min
                        PSpec = 1045, # Pa
                        LifeSpec = 40000,
                        Sampling = Sampling,
                        NumSample = NumSample)

fan120 = sample_generator(RpmSpec = 7650,
                        PowerSpec = 1.3 * 48,
                        TempSpec = 40,
                        DiameterSpec = 120,
                        ThicknessSpec = 38,
                        QSpec = 7.49, # m^3/min
                        PSpec = 532.5, # Pa,
                        LifeSpec = 40000,
                        Sampling = Sampling,
                        NumSample = NumSample)
# 各fan統合
list = fan40
for data in fan40cr:
    list.append(data)

for data in fan120:
    list.append(data)

df = pd.DataFrame(list,    # listは3種ファン統合。別々にやる場合は、fan40, fan40cr or fan120
                 columns=['sample_id',
                          'defect',
                          'time', 
                          'rpm', 
                          'temp', 
                          'power', 
                          'cum_rpm', 
                          'cum_temp', 
                          'cum_power', 
                          'RpmSpec', 
                          'PowerSpec', 
                          'DiameterSpec', 
                          'ThicknessSpec', 
                          'QSpec', 
                          'PSpec', 
                          'LifeSpec',
                          'cumurated_life_impact_factor', 
                          'death',
                          'k',
                          'remaining_life'])

# df.to_csv("./sample_data_check3.csv")

print('df= ', df.info())

fig, ax = plt.subplots()
ax.scatter(df['time'], df['rpm'], c=df['sample_id'], s=10, alpha=0.5)
plt.xlabel('time [H]',size=12)
plt.ylabel('rpm',size=12)

fig, ax = plt.subplots()
ax.scatter(df['time'], df['power'], c=df['sample_id'])
plt.xlabel('time [H]',size=12)
plt.ylabel('power',size=12)

fig, ax = plt.subplots()
ax.scatter(df['time'], df['remaining_life'], c=df['sample_id'])
plt.xlabel('time [H]',size=12)
plt.ylabel('remaining_life [H]',size=12)

fig, ax = plt.subplots()
ax.scatter(df['time'], df['k'], c=df['sample_id'])
plt.xlabel('time [H]',size=12)
plt.ylabel('k (Arrhenius coefficient)',size=12)

fig, ax = plt.subplots()
ax.scatter(df['time'], df['cum_rpm'], c=df['sample_id'])
plt.xlabel('time [H]',size=12)
plt.ylabel('cum_rpm',size=12)

fig, ax = plt.subplots()
ax.scatter(df['time'], df['cum_power'], c=df['sample_id'])
plt.xlabel('time [H]',size=12)
plt.ylabel('cum_power',size=12)

fig, ax = plt.subplots()
ax.scatter(df['time'], df['cum_temp'], c=df['sample_id'])
plt.xlabel('time [H]',size=12)
plt.ylabel('cum_temp',size=12)

fig, ax = plt.subplots()
ax.scatter(df['time'], df['cumurated_life_impact_factor'], c=df['sample_id'])
plt.xlabel('time [H]',size=12)
plt.ylabel('cumurated_life_impact_factor',size=12)

# Corelation Analysis

sns.pairplot(df.loc[: ,'defect':'cumurated_life_impact_factor'])


# %%
df = df.dropna(how="any")

indexNames = df[ df['death'] == 1 ].index
df.drop(indexNames , inplace=True)

indexNames = df[ df['remaining_life'] == 0 ].index
df.drop(indexNames , inplace=True)

df.to_csv('./sample_data_v02.csv')
df

# %% [markdown]
# ### Prediction

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import seaborn as sns

df = pd.read_csv('sample_data_v02.csv', index_col=[0])
df = df.dropna(how="any")

# print(df.head(), df.tail())
df.info()
# print(df.describe())


# %%
plt.figure(figsize=(8, 8))
plt.hist(df['remaining_life'], bins=100)
plt.title('Data Histgram for prediction value')
plt.xlabel('remaining_life',size=12)
plt.show()


# %%
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 機械学習では、学習させる特徴量をX, 求める答えをyで表す。
X = df.drop(columns=['remaining_life', 'sample_id', 'defect', 'k'])
y = df['remaining_life']

print(X.shape)
print(y.shape)

# trainとは学習に用いるデータ、testとは検証用に取っておくまだ見ぬデータ。
# 自動的に検証用のまだ見ぬデータを20%とっておく。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
print(X_train.shape)
print(X_test.shape)


# %%

params = {
    'silent': 1,
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': 0.1,
    'tree_method': 'exact',
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'predictor': 'cpu_predictor'
}

# Xgboost params tutorial
# https://qiita.com/FJyusk56/items/0649f4362587261bd57a

## objective
# reg:linear(線形回帰)
# reg:logistic(ロジスティック回帰)
# binary:logistic(2項分類で確率を返す)
# multi:softmax(多項分類でクラスの値を返す)

## eval_metric
# rmse(2乗平均平方根誤差)
# logloss(負の対数尺度)
# error(2-クラス分類のエラー率)
# merror(多クラス分類のエラー率)
# mlogloss(多クラスの対数損失)
# auc(ROC曲線下の面積で性能の良さを表す)
# mae(平均絶対誤差)



# GPUの場合
# params = {
#     'silent': 1,
#     'max_depth': 6,
#     'min_child_weight': 1,
#     'eta': 0.1,
#     'tree_method': 'gpu_exact',
#     'objective': 'gpu:reg:linear',
#     'eval_metric': 'rmse',
#     'predictor': 'gpu_predictor'
# }

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
model = xgb.train(params=params,
                  dtrain=dtrain,
                  num_boost_round=1000,
                  early_stopping_rounds=5,
                  evals=[(dtest, 'test')])


# %%
print(model)
model.save_model('./xgb1.model')

model.load_model('./xgb1.model')

prediction = model.predict(xgb.DMatrix(X_test), 
                           ntree_limit=model.best_ntree_limit)

plt.figure(figsize=(8, 8))
# plt.scatter(y_test[:1000], prediction[:1000], alpha=0.2)
plt.scatter(y_test, prediction, alpha=0.2)
plt.title('Evaluation between y_test=Correct Answer and Prediction. if gradient is 1, perfect')
plt.xlabel('Correct Answer',size=12)
plt.ylabel('Prediction',size=12)
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
xgb.plot_importance(model, max_num_features=12, height=0.8, ax=ax)
plt.show()


# %%



# %%
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
prediction_df = pd.DataFrame(prediction, columns=['remaining_life_pred']) 

X_test_df_reset = X_test_df.reset_index()
y_test_df_reset = y_test_df.reset_index()
prediction_df_reset = prediction_df.reset_index()

print(y_test_df_reset)

y_test_list = y_test_df_reset['remaining_life'].to_list()
prediction_list = prediction_df_reset['remaining_life_pred'].to_list()

print(y_test_list[:5], len(y_test_list))
print(prediction_list[:5], len(prediction_list))
difference_list = []

for i in range(len(y_test_list)):
    diff = (y_test_list[i] - prediction_list[i]) / y_test_list[i]
    difference_list.append(diff)

print(difference_list[:5])

difference_df = pd.DataFrame(difference_list, columns=['defference_rate'])
print(difference_df)
print('')
print('difference_df.info()==>', difference_df.info())
print('difference_df.describe()==>', difference_df.describe())
print('length comparison==>', len(X_test_df_reset), len(y_test_df_reset), len(prediction_df_reset), len(difference_df))

difference_df_reset = difference_df.reset_index()

plt.figure(figsize=(8, 8))
plt.hist(difference_df['defference_rate'], bins=100, range=(-1,1))
plt.title('Accuracy Verification (x=0 is correct)')
plt.show()

report_df = pd.concat([X_test_df_reset, y_test_df_reset, prediction_df_reset, difference_df_reset], axis=1)

report_df

report_df.to_csv("./report_remaining_life.csv")


# %%



# %%


# %% [markdown]
# #### Can Defects be detected?

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import seaborn as sns

df = pd.read_csv('sample_data_v02.csv', index_col=[0])
df = df.dropna(how="any")

# print(df.head(), df.tail())
df.info()
# print(df.describe())


# %%
plt.figure(figsize=(8, 8))
plt.hist(df['defect'], bins=2)
plt.title('Data Histgram for prediction value')
plt.xlabel('defect (0; good, 1; defect)',size=12)
plt.show()


# %%
from sklearn.model_selection import train_test_split
import xgboost as xgb

X = df.drop(columns=['remaining_life', 'sample_id', 'defect', 'k'])
y = df['defect']

print(X.shape)
print(y.shape)

# trainとは学習に用いるデータ、testとは検証用に取っておくまだ見ぬデータ。
# 自動的に検証用のまだ見ぬデータを20%とっておく。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
print(X_train.shape)
print(X_test.shape)


# %%
params = {
    'silent': 1,
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': 0.1,
    'tree_method': 'exact',
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'predictor': 'cpu_predictor'
}

# GPUの場合
# params = {
#     'silent': 1,
#     'max_depth': 6,
#     'min_child_weight': 1,
#     'eta': 0.1,
#     'tree_method': 'gpu_exact',
#     'objective': 'gpu:reg:linear',
#     'eval_metric': 'rmse',
#     'predictor': 'gpu_predictor'
# }

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
model = xgb.train(params=params,
                  dtrain=dtrain,
                  num_boost_round=1000,
                  early_stopping_rounds=5,
                  evals=[(dtest, 'test')])


# %%
print(model)
model.save_model('./xgb1.model')

model.load_model('./xgb1.model')

prediction = model.predict(xgb.DMatrix(X_test), 
                           ntree_limit=model.best_ntree_limit)

plt.figure(figsize=(8, 8))
# plt.scatter(y_test[:1000], prediction[:1000], alpha=0.2)
plt.scatter(y_test, prediction, alpha=0.2)
plt.title('Evaluation between y_test=Correct Answer and Prediction')
plt.xlabel('Correct Answer',size=12)
plt.ylabel('Prediction',size=12)
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
xgb.plot_importance(model, max_num_features=12, height=0.8, ax=ax)
plt.show()


# %%
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
prediction_df = pd.DataFrame(prediction, columns=['defect_pred']) 

X_test_df_reset = X_test_df.reset_index()
y_test_df_reset = y_test_df.reset_index()
prediction_df_reset = prediction_df.reset_index()

print(y_test_df_reset)

y_test_list = y_test_df_reset['defect'].to_list()
prediction_list = prediction_df_reset['defect_pred'].to_list()

print(y_test_list[:5], len(y_test_list))
print(prediction_list[:5], len(prediction_list))
difference_list = []

for i in range(len(y_test_list)):
    diff = (y_test_list[i] - prediction_list[i])
    difference_list.append(diff)

print(difference_list[:5])

difference_df = pd.DataFrame(difference_list, columns=['defference_rate'])
print(difference_df)
print('')
print('difference_df.info()==>', difference_df.info())
print('difference_df.describe()==>', difference_df.describe())
print('length comparison==>', len(X_test_df_reset), len(y_test_df_reset), len(prediction_df_reset), len(difference_df))

difference_df_reset = difference_df.reset_index()

plt.figure(figsize=(8, 8))
plt.hist(difference_df['defference_rate'], bins=100, range=(-1,1))
plt.title('Accuracy Verification (x=0 is correct)')
plt.show()

report_df = pd.concat([X_test_df_reset, y_test_df_reset, prediction_df_reset, difference_df_reset], axis=1)

report_df

report_df.to_csv("./report_detect_defect.csv")


# %%
print('Completed!!Completed!!!Completed!!!!Completed!!!!Completed!!!!Completed!!!!Completed!!!')


# %%



# %%



