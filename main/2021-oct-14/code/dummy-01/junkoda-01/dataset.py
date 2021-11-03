from common import *
from ventilator import *

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
#https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook

#---
data_dir = root_dir+'/data'


i_col=[
    'id',
    'breath_id',
]
x_col=[
    # 'time_step',
    # 'u_in',
    # 'u_out',
    # 'R',
    # 'C',
    'time_step', 'u_in', 'u_out', 'area', 'u_in_cumsum', 'u_in_lag2','u_in_lag4',
    'r_20', 'r_5', 'r_50', 'c_10', 'c_20', 'c_50',
    'ewm_u_in_mean', 'ewm_u_in_std', 'ewm_u_in_corr', 'rolling_10_mean',
    'rolling_10_max', 'rolling_10_std', 'expand_mean', 'expand_max',
    'expand_std'
]
y_col=[
    'pressure',
]




# https://www.kaggle.com/soumenksarker/bi-lstm-model-pressure-predict-gpu-infer
def make_df(mode='train'):
    if mode=='train':
        df = pd.read_csv(data_dir + '/train.csv')
    if mode=='test':
        df = pd.read_csv(data_dir + '/test.csv')
        df.loc[:,'pressure']=-1

    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['u_in_lag4'] = df['u_in'].shift(4).fillna(0)

    df['r'] = df['R'].astype(str)
    df['c'] = df['C'].astype(str)
    df = pd.get_dummies(df, drop_first = False)

    g = df.groupby('breath_id')['u_in']
    df['ewm_u_in_mean'] = g.ewm(halflife=10).mean().reset_index(level=0, drop=True)
    df['ewm_u_in_std'] = g.ewm(halflife=10).std().reset_index(level=0, drop=True)
    df['ewm_u_in_corr'] = g.ewm(halflife=10).corr().reset_index(level=0, drop=True)

    df['rolling_10_mean'] = g.rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    df['rolling_10_max'] = g.rolling(window=10, min_periods=1).max().reset_index(level=0, drop=True)
    df['rolling_10_std'] = g.rolling(window=10, min_periods=1).std().reset_index(level=0, drop=True)

    df['expand_mean'] = g.expanding(2).mean().reset_index(level=0, drop=True)
    df['expand_max'] = g.expanding(2).max().reset_index(level=0, drop=True)
    df['expand_std'] = g.expanding(2).std().reset_index(level=0, drop=True)
    df = df.fillna(0)


    #---
    print('df.shape', df.shape)
    print(df.columns)
    return df


def make_fold(df, mode='train-1', ):
    if 'train' in mode:
        fold = int(mode[-1])
        kf = KFold(n_splits=5, random_state=123, shuffle=True)
        train_idx, valid_idx = [],[]
        for t, v in kf.split(df['breath_id'][::80]):
            train_idx.append(t)
            valid_idx.append(v)
        return train_idx[fold], valid_idx[fold]

    if 'test' in mode:
        valid_idx = np.arange(len(df)//80)
        return valid_idx


class VentilatorDataset(Dataset):
    def __init__(self, df, idx, scaler):
        super().__init__()
        self.length = len(idx)
        self.idx = idx
        #self.feature   = df[x_col].values.astype(np.float32).reshape(-1, 80, len(x_col))

        feature = scaler.transform(df[x_col].values).astype(np.float32)
        self.feature   = feature.reshape(-1, 80, len(x_col))
        self.pressure  = df[y_col].values.astype(np.float32).reshape(-1,80)
        self.u_out     = df['u_out'].values.astype(np.float32).reshape(-1,80)
        self.breath_id = df['breath_id'].values[::80]
        zz=0


    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        i = self.idx[index]
        r = {
            'index'     : index,
            'breath_id' : self.breath_id[i],
            'u_out'     : self.u_out[i],
            'pressure'  : self.pressure[i],
            'feature'   : self.feature[i],
        }
        return r



#########################################################


def run_check_dataset():

    df = make_df(mode='train')
    print('df', df.shape)
    #---

    scaler = RobustScaler()
    scaler.fit_transform(df[x_col])

    #---
    train_idx, valid_idx = make_fold(df, mode='train-1')
    dataset = VentilatorDataset(df, scaler, train_idx)
    print(dataset)

    for i in range(5):
        #i = np.random.choice(len(dataset))#272 #
        r = dataset[i]

        print('---')
        print('index     :', r['index'])
        print('breath_id    :', r['breath_id'])
        print('pressure  :', r['pressure'].shape)
        print('u_out     :', r['u_out'].shape)
        print('feature   :', r['feature'].shape)

        if 0:
            plt.clf()
            plt.plot(r['feature' ], c='gray')
            plt.plot(r['pressure'], label='pressure', c='red')
            plt.legend()
            plt.waitforbuttonpress()
            #plt.show()

        if 1:
            j = train_idx[i]
            d = df.iloc[j*80:(j+1)*80]
            d0 = df.iloc[j*80]

            print('breath_id  :', d0['breath_id'])
            print('R :', d0['R'])
            print('C :', d0['C'])

            #---
            plt.clf()
            plt.plot(d['time_step'],d['pressure'], label='pressure',c='red')
            plt.plot(d['time_step'],d['u_in'], label='u_in',c='green')
            plt.plot(d['time_step'],d['u_out'], label='u_out',c='blue')

            # p = d['pressure'].values
            # u_in = d['u_in'].values
            # plt.plot(p[1:]-p[:-1], label='pressure')
            # plt.plot(u_in[1:]-u_in[:-1], label='u_in')

            plt.legend()
            plt.waitforbuttonpress(1)
            #plt.show()


    loader = DataLoader(
        dataset,
        sampler = RandomSampler(dataset),
        batch_size  = 8,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = True,
    )
    for t,batch in enumerate(loader):
        if t>30: break

        print(t, '-----------')
        print('index : ', batch['index'])
        print('breath_id : ', batch['breath_id'])
        print('u_out : ')
        print('\t', batch['u_out'].shape, batch['u_out'].is_contiguous())
        print('pressure : ')
        print('\t', batch['pressure'].shape, batch['pressure'].is_contiguous())
        print('feature : ')
        print('\t', batch['feature'].shape, batch['feature'].is_contiguous())
        print('')


##################################################################################
if __name__ == '__main__':
    run_check_dataset()

