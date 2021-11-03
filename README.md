# Kaggle_Google-Brain-Ventilator-Pressure-Prediction

# Description
患者が呼吸困難に陥ったとき、医師は何をするのだろうか。人工呼吸器を使って、鎮静状態の患者の肺に気管から酸素を送り込むのだ。しかし、人工呼吸は臨床医の手が必要であり、その限界はCOVID-19パンデミックの初期に顕著に現れました。同時に、機械式人工呼吸器を制御する新しい方法を開発するには、臨床試験に至るまでに莫大な費用がかかります。高品質のシミュレータは、この障壁を減らすことができます。

現在のシミュレータは、各モデルが単一の肺の設定をシミュレートするアンサンブルとしてトレーニングされています。しかし、肺とその属性は連続した空間を形成しているため、患者の肺の違いを考慮したパラメトリックなアプローチを検討する必要があります。

Google Brainのチームは、プリンストン大学と提携し、人工呼吸制御のための機械学習をめぐるコミュニティの発展を目指しています。ニューラルネットワークと深層学習は、現在の業界標準であるPID制御よりも、さまざまな特性を持つ肺をよりよく一般化できると考えています。

このコンペティションでは、鎮静状態の患者の肺に接続された人工呼吸器のシミュレーションを行います。優秀な作品は、肺の特性であるコンプライアンスと抵抗を考慮しています。

成功すれば、機械式人工呼吸器を制御する新しい方法を開発する際のコスト面での障壁を克服することができます。これにより、患者に適応したアルゴリズムへの道が開かれ、この斬新な時代とそれ以降の時代において、臨床医の負担を軽減することができます。その結果、患者さんの呼吸を助けるための人工呼吸器の治療法がより広く普及することになるかもしれません。

# Evaluation
競技は、各呼吸の吸気段階における予測圧力と実際の圧力の間の平均絶対誤差として採点されます。呼気相は採点されない。スコアは以下のように与えられる。

$$|𝑋−𝑌|$$
ここで、𝑋は予測圧力のベクトル、𝑌はテストセットの全呼吸における実際の圧力のベクトルである。

## 提出ファイル
テストセットの各idについて、圧力変数の値を予測する必要があります。ファイルはヘッダーを含み、以下の形式で作成してください。

```
id,pressure
1,20
2,23
3,24
etc.
```

# Data
本大会で使用した人工呼吸器のデータは、オープンソースの人工呼吸器を改造し、呼吸回路を介して人工蛇腹試験肺に接続して作成しました。下図はその設定を示したもので、2つの制御入力を緑で、予測する状態変数（気道圧）を青で示しています。1つ目の制御入力は，0〜100の連続変数で，空気を肺に入れるために吸気電磁弁を開く割合を表します（すなわち，0は完全に閉じて空気を入れず，100は完全に開きます）。2つ目の制御入力は、空気を出すための探索電磁弁が開いている（1）か閉じている（0）かを表す二値変数です。

この競技では，参加者に多数の呼吸の時系列が与えられ，制御入力の時系列が与えられた場合に，呼吸時の呼吸回路内の気道圧力を予測することを学習する．  
<img width="732" alt="Screen Shot 2021-09-30 at 22 18 18" src="https://user-images.githubusercontent.com/71954051/135462647-b02c2f5d-522e-4987-89db-aa64dbb8c1dd.png">

各時系列は、約3秒間の呼吸を表しています。ファイルは、各行が呼吸のタイムステップとなるように構成されており、後述する2つの制御信号、その結果としての気道圧力、および肺の関連属性を示しています。

## Files
- train.csv - トレーニングセット
- test.csv - テストセット
- sample_submission.csv - 正しいフォーマットで作成されたサンプル投稿ファイル

## Columns
- breath_id - グローバルに一意な呼吸のタイムステップ
- R - 気道がどの程度制限されているかを示す肺属性（単位：cmH2O/L/S）。物理的には、流量（時間当たりの空気量）の変化に対する圧力の変化です。直感的には、ストローで風船を膨らませるようなイメージです。ストローの直径を変えることでRを変化させることができ、Rが大きいほど吹きにくくなります。
- C - 肺の適合性を示す肺属性（単位：mL/cmH2O）。物理的には、圧力の変化に対する体積の変化を表します。直感的には、同じ風船の例を想像してください。風船のラテックスの厚さを変えることでCを変化させることができます。Cが大きいほどラテックスが薄く、吹きやすくなります。
- time_step - 実際のタイムスタンプです。
- u_in - 吸気ソレノイドバルブの制御入力です。0～100の範囲で設定できます。
- u_out - 探索用ソレノイドバルブの制御入力です。0または1のいずれかです。
- pressure - 呼吸回路で測定された気道の圧力で、単位はcmH2Oです。

# Log
## 20210930
- join
- LSTM実装できるようにしたい

## 20211001
### GB_nb000
- [Deep Learning Starter : Simple LSTM](https://www.kaggle.com/theoviel/deep-learning-starter-simple-lstm)写経
- breath_idごとにinput, outputを設定

## 20211002, 1003
- 昨日の続き
- 写経終了
- コードは30%くらい理解
- LSTMの理論勉強しなきゃ

## 20211012
### GB_nb002
- [reject]特徴量追加
    - div_R_C
    - div_C_R
    - R, C ; str　→ R, C ; float
- 0.70237, 0.64347, 0.69892, 0.63648, 0.68166, 0.71558, 0.69029, 0.72464, 0.70306, 0.66260
- 0.685907

- 公開コード
- 0.70344, 0.64502, 0.69168, 0.65081, 0.70669, 0.72130, 0.64494, 0.71879, 0.69937, 0.63291
- 0.681492

- [reject]input_dim=1024→2048, 5層
- fold9 val_loss=0.18

- 5fold
- 公開code
- 0.75152, 0.74191, 0.72600, 0.69858, 0.66520
- 0.716642

- lag5追加(debug)
- 0.69271, 0.68639, 0.70154, 0.68344, 0.70840
- 0.694496

- lag6追加
- 0.72801, 0.70793, 0.74417

## 20211014
### 公開code;: gb-nb003
- baseline
- 5fold
- debug=True
- seed=71
- fold: 0.70642, 0.70791, 0.71608, 0.67396, 0.72496
- cv: 0.705866


### gb-nb004
- 5fold
- debug=True
- seed=71
- add: last_value_u_in
- fold: 0.73872, 0.68623, 0.68638, 0.66329, 0.63300
- cv: 0.681524


### gb-ng002
- 5fold
- debug=False
- add: lag5

### gb-nb004
- 5fold
- debug=True
- seed=71
- add: lag5, last_value_u_in
- fold: 0.71884, 0.70275, 0.69137, 0.75560, 0.64964
- cv: 0.70364

### gb-nb004
- 5fold
- debug=False
- seed=71
- add: last_value_u_in
- fold: 
- cv:
- lb: 

## 20211019
- メモサボってた
### gb-nb008
- gb-nb007をclassifierにする 

## 20211020
### gb-nb008
-[GB - VPP - Whoppity dub dub](https://www.kaggle.com/dlaststark/gb-vpp-whoppity-dub-dub)の特徴量からoutに関するものを引いた
- 
```
def add_features(df):
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()


    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)

    df = df.fillna(0)
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] = df['u_in_cumsum'] / df['count']
    
    df['breath_id_lag'] = df['breath_id'].shift(1).fillna(0)
    df['breath_id_lag2'] = df['breath_id'].shift(2).fillna(0)
    df['breath_id_lagsame'] = np.select([df['breath_id_lag']==df['breath_id']],[1],0)
    df['breath_id_lag2same'] = np.select([df['breath_id_lag2']==df['breath_id']], [1], 0)
    df['breath_id__u_in_lag'] = df['u_in'].shift(1).fillna(0)
    df['breath_id__u_in_lag'] = df['breath_id__u_in_lag'] * df['breath_id_lagsame']
    df['breath_id__u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['breath_id__u_in_lag2'] = df['breath_id__u_in_lag2'] * df['breath_id_lag2same']


    df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['ewm_u_in_mean'] = (df\
                           .groupby('breath_id')[['u_in']]\
                           .ewm(halflife=9)\
                           .mean()\
                           .reset_index(level=0,drop=True))
    df[['15_in_sum', '15_in_min', '15_in_max', '15_in_mean']] = (df\
                                                                 .groupby('breath_id')['u_in']\
                                                                 .rolling(window=15, min_periods=1)\
                                                                 .agg({'15_in_sum':'sum',
                                                                       '15_in_min':'min',
                                                                       '15_in_max':'max',
                                                                       '15_in_mean':'mean'})\
                                                                 .reset_index(level=0, drop=True))
    

    df['u_in_lagback_diff1'] = df['u_in'] - df['u_in_lag_back1']
    df['u_in_lagback_diff2'] = df['u_in'] - df['u_in_lag_back2']
    

    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)

    return df

train = add_features(train_df)
test = add_features(test_df)
del train_df, test_df
gc.collect()
```

### gb-nb008
- よくある特徴量使ってる
- 

### gb-nb011
- cv: 0.165083407259815, 0.15780835227398698, 0.16185144929734957, 0.1552711538175763, 0.15557388985375448, 0.15929822242330105, 0.15952012520386283, 0.1571477531343505, 0.15855710421709027
- 0.1589


### gb-nb015
- make fft features
    - abs(log(features))
    - features = fft, fft_w, fft_cos, fft_cos_w

### datasets/train(test)_fft.pkl
- abs(log(features))
- features = fft, fft_w, fft_cos, fft_cos_w

### datasets/train(test)_fft_clip.pkl
- abs(log(features))
- features = fft, fft_w, fft_cos, fft_cos_w
- clipped
    - df['fft'] = df['fft'].clip(0,10)
    - df['fft_w'] = df['fft_w'].clip(0,15)
    - df['fft_cos'] = df['fft_cos'].clip(0,20)
    - df['fft_cos_w'] = df['fft_cos_w'].clip(0,25)


## 20211027
### gb-nb016
- 0.1593, 0.1615, 0.1581, 0.1624, 0.1569, 0.1562, 0.1581, 0.1588, 0.1579, 0.1578
- cv: 0.1587

## 20211028
### datasets/train[test]_fft_physics.pkl
```
def add_features(df):
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()


    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)

    df = df.fillna(0)
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] = df['u_in_cumsum'] / df['count']
    
    df['breath_id_lag'] = df['breath_id'].shift(1).fillna(0)
    df['breath_id_lag2'] = df['breath_id'].shift(2).fillna(0)
    df['breath_id_lagsame'] = np.select([df['breath_id_lag']==df['breath_id']],[1],0)
    df['breath_id_lag2same'] = np.select([df['breath_id_lag2']==df['breath_id']], [1], 0)
    df['breath_id__u_in_lag'] = df['u_in'].shift(1).fillna(0)
    df['breath_id__u_in_lag'] = df['breath_id__u_in_lag'] * df['breath_id_lagsame']
    df['breath_id__u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['breath_id__u_in_lag2'] = df['breath_id__u_in_lag2'] * df['breath_id_lag2same']


    df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['ewm_u_in_mean'] = (df\
                           .groupby('breath_id')[['u_in']]\
                           .ewm(halflife=9)\
                           .mean()\
                           .reset_index(level=0,drop=True))
    df[['15_in_sum', '15_in_min', '15_in_max', '15_in_mean']] = (df\
                                                                 .groupby('breath_id')['u_in']\
                                                                 .rolling(window=15, min_periods=1)\
                                                                 .agg({'15_in_sum':'sum',
                                                                       '15_in_min':'min',
                                                                       '15_in_max':'max',
                                                                       '15_in_mean':'mean'})\
                                                                 .reset_index(level=0, drop=True))
    

    df['u_in_lagback_diff1'] = df['u_in'] - df['u_in_lag_back1']
    df['u_in_lagback_diff2'] = df['u_in'] - df['u_in_lag_back2']
    

    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)

    df['exponent']=(- df['time_step'])/(df['R']*df['C'])
    df['factor']=np.exp(df['exponent'])
    df['vf']=(df['u_in_cumsum']*df['R'])/df['factor']
    df['vt']=0
    df.loc[df['time_step'] != 0, 'vt']=df['area']/(df['C']*(1 - df['factor']))
    df['v']=df['vf']+df['vt']

    return df
Index(['time_step', 'u_in', 'u_out', 'pressure', 'cross', 'cross2', 'area',
       'time_step_cumsum', 'u_in_cumsum', 'u_in_lag1', 'u_in_lag_back1',
       'u_in_lag2', 'u_in_lag_back2', 'u_in_lag3', 'u_in_lag_back3',
       'u_in_lag4', 'u_in_lag_back4', 'breath_id__u_in__max',
       'breath_id__u_out__max', 'breath_id__u_in__diffmax',
       'breath_id__u_in__diffmean', 'u_in_diff1', 'u_in_diff2', 'u_in_diff3',
       'u_in_diff4', 'u_in_cummean', 'breath_id__u_in_lag',
       'breath_id__u_in_lag2', 'time_step_diff', 'ewm_u_in_mean', '15_in_sum',
       '15_in_min', '15_in_max', '15_in_mean', 'u_in_lagback_diff1',
       'u_in_lagback_diff2', 'R_20', 'R_5', 'R_50', 'C_10', 'C_20', 'C_50',
       'R__C_20__10', 'R__C_20__20', 'R__C_20__50', 'R__C_50__10',
       'R__C_50__20', 'R__C_50__50', 'R__C_5__10', 'R__C_5__20', 'R__C_5__50',
       'fft', 'fft_w', 'fft_cos', 'fft_cos_w', 'exponent', 'factor', 'vf',
       'vt', 'v'],
      dtype='object')
```

### nb021
- アンサンブル用

### gb-nb019
```
col = ['u_in_diff1', 
       'u_in_diff2', 
       'u_in', 
       'u_in_lag1', 
       'u_in_lag_back1',
       'u_in_lag_back2',
       'u_in_cumsum',
       ]
deviation, zscore, median

特徴量75個
```
- メモリエラーと特徴量の数
    - 75セーフ
    - 87アウト
    - 82アウト

### gb-nb021
```
col = ['u_in_diff1', 
       'u_in_diff2', 
       'u_in_diff3', 
       'u_in', 
       'u_in_lag1', 
       'u_in_cumsum',
       ]
deviation, zscore, std
```
- 15fold

### gb-nb020
- gb-nb016のモデルの活性化関数を selu→gelu にした
- 0.1581, fold2がうまく行かなかった

### gb-nb022
- 横にGRUから横にLSTM layer2層増やした

### gb-nb023
- GRUを全てLSTMに変更
- Multiply → Average(うまく行かない気がする)(うまく行ったわ)
- 0.1585, 0.1609, 0.1593, 0.1595, 0.1555, 0.1545, 0.1570, 0.1608, 0.1567, 0.1598
- 0.15825
- batch_size = 512
- seed = 71
- fold = 10

### gb-nb024
- Average + GRU
- 0.1597, 0.1652, 0.1601, 0.1577

### gb-nb028[旧]
- from nb023
- batch_size = 128
- seed = 771
- fold = 10
- 0.1621



### gb-nb026[WIP]
- from nb023
- batch_size: 256
- seed = 72
- fold = 15
- 0.1544, 0.1595, 0.1590, 0.1573, 0.1561, 0.1538

### gb-nb028[WIP]
- batch_size = 512
- seed = 771
- fold = 15
- 0.1597, 0.1572, 0.1563, 0.1599

### gb-nb032[WIP]
- from nb023
- batch_size = 256
- seed = 7
- fold = 10
- 0.1581, 0.1600, 0.1590, 0.1567, 0.1581, 0.1582

### gb-kaggle-nb016
- batch_size = 512
- seed = 700
- fold = 15
- 
