import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class CMAPSS(Dataset):
    def __init__(self, data_path, normalization_type, win_len, diff_order, train_mode):
        self.data_path = data_path
        self.normalization_type = normalization_type
        self.win_len = win_len
        self.diff_order = diff_order
        self.train_mode = train_mode
        if self.train_mode == 'test':
            anno_pd = prepare_test_data(self.data_path, self.normalization_type, self.win_len, self.diff_order)
            self.data = anno_pd['data'].tolist()
            self.cycle = anno_pd['cycle'].tolist()
            self.work_condition = anno_pd['work_condition'].tolist()
        elif self.train_mode == 'val':
            anno_pd = prepare_val_data(self.data_path, self.normalization_type, self.win_len, self.diff_order)
            self.data = anno_pd['data'].tolist()
            self.label = anno_pd['label'].tolist()
            self.cycle = anno_pd['cycle'].tolist()
            self.work_condition = anno_pd['work_condition'].tolist()
        else:
            anno_pd = prepare_train_data(self.data_path, self.normalization_type, self.win_len, self.diff_order)
            self.data = anno_pd['data'].tolist()
            self.label = anno_pd['label'].tolist()
            self.cycle = anno_pd['cycle'].tolist()
            self.work_condition = anno_pd['work_condition'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.train_mode == 'test':
            img = self.data[item]
            if self.diff_order == 0:
                img = np.squeeze(img)
                img = img.transpose((1, 0))
            # img = img.transpose((2, 1, 0))
            img = img.astype(np.float32)
            cycle = self.cycle[item]
            cycle = np.array(cycle, dtype='float32')
            work_condition = self.work_condition[item]
            work_condition = work_condition.astype(np.float32)
            return img, work_condition, cycle
        else:
            img = self.data[item]
            if self.diff_order == 0:
                img = np.squeeze(img)
                img = img.transpose((1, 0))
            #img = img.transpose((2, 1, 0))
            img = img.astype(np.float32)
            label = self.label[item]
            label = np.array(label, dtype='float32')
            cycle = self.cycle[item]
            cycle = np.array(cycle, dtype='float32')
            work_condition = self.work_condition[item]
            work_condition = work_condition.astype(np.float32)
            return img, label, work_condition, cycle


def prepare_adj(data_path):
    am = pd.read_csv(data_path[3], header=None)
    am = am.values
    normed_am = am / (am.sum(axis=0) + 1e-18)
    return normed_am


def prepare_tadj(data_path):
    am = pd.read_csv(data_path, header=None)
    am = am.values
    normed_am = am / (am.sum(axis=0) + 1e-18)
    return normed_am


def prepare_train_data(data_path, normalization_type, win_len, diff_order):
    # Read the data
    df_train = local_load_data(data_path[0])
    # Normalization
    # s_cols = ['C1', 'C2', 'C3'] + [x for x in df_train.columns if 'S' in x] # for FD002 and FD004
    s_cols = [x for x in df_train.columns if 'S' in x]  # for FD001 and FD003
    df_train = normalization_train_data(df_train, s_cols, normalization_type)
    c_cols = ['C1', 'C2', 'C3']
    df_train = normalization_train_data(df_train, c_cols, normalization_type)
    # Add RUL
    df_train = add_rul(df_train, total_rul=125, end_point=np.zeros(df_train['ID'].max()))
    # Generate Training examples
    data_train = generate_sequences(df_train, win_len, diff_order, s_cols, c_cols)
    return data_train


def prepare_val_data(data_path, normalization_type, win_len, diff_order):
    # Read the data
    df_train = local_load_data(data_path[0])
    df_test = local_load_data(data_path[1])
    test_rul = np.loadtxt(data_path[2])
    # Normalization
    # s_cols = ['C1', 'C2', 'C3'] + [x for x in df_train.columns if 'S' in x]   # for FD002 and FD004
    s_cols = [x for x in df_train.columns if 'S' in x]      # FD001 and FD003
    df_test = normalization_test_data(df_train, df_test, s_cols, normalization_type)
    c_cols = ['C1', 'C2', 'C3']
    df_test = normalization_test_data(df_train, df_test, c_cols, normalization_type)
    # Add RUL
    df_test = add_rul(df_test, total_rul=125, end_point=test_rul)
    # Generate validation examples
    data_validation = generate_val_sequences(df_test, win_len, diff_order, s_cols, c_cols)
    return data_validation


def prepare_test_data(data_path, normalization_type, win_len, diff_order):
    # Read the data
    df_train = local_load_data(data_path[0])
    df_test = local_load_data(data_path[1])
    test_rul = np.loadtxt(data_path[2])
    # Normalization
    # s_cols = ['C1', 'C2', 'C3'] + [x for x in df_train.columns if 'S' in x]
    s_cols = [x for x in df_train.columns if 'S' in x]
    df_test = normalization_test_data(df_train, df_test, s_cols, normalization_type)
    c_cols = ['C1', 'C2', 'C3']
    df_test = normalization_test_data(df_train, df_test, c_cols, normalization_type)
    # Add RUL
    df_test = add_rul(df_test, total_rul=125, end_point=test_rul)
    # Generate testing examples
    data_test = generate_sequences(df_test, win_len, diff_order, s_cols, c_cols)
    return data_test


def normalization_train_data(df, cols, type):
    for j in range(len(cols)):
        if type == '-1-1':
            min_value = df.loc[:, cols[j]].min()
            max_value = df.loc[:, cols[j]].max()
            df.loc[:, cols[j]] = 2*(df.loc[:, cols[j]] - min_value) / (max_value - min_value)-1
        elif type == 'z-score':
            mean_value = df.loc[:, cols[j]].mean()
            std_value = df.loc[:, cols[j]].std()
            df.loc[:, cols[j]] = (df.loc[:, cols[j]] - mean_value) / std_value
        else:
            raise NameError('This normalization is not included!')
    return df


def smooth_data(df, cols, win_len=30):
    engine_number = df['ID'].max()
    for i in range(1, engine_number+1):
        index = df[df['ID'] == i].index
        for j in range(len(cols)):
            df.loc[index, cols[j]] = df.loc[index, cols[j]].rolling(win_len, min_periods=1).mean()
    return df


def normalization_test_data(df_train, df_test, cols, type):
    for j in range(len(cols)):
        if type == '-1-1':
            min_value = df_train.loc[:, cols[j]].min()
            max_value = df_train.loc[:, cols[j]].max()
            df_train.loc[:, cols[j]] = 2*(df_train.loc[:, cols[j]] - min_value) / (max_value - min_value)-1
            df_test.loc[:, cols[j]] = 2*(df_test.loc[:, cols[j]] - min_value) / (max_value - min_value)-1
        elif type == 'z-score':
            mean_value = df_train.loc[:, cols[j]].mean()
            std_value = df_train.loc[:, cols[j]].std()
            df_train.loc[:, cols[j]] = (df_train.loc[:, cols[j]] - mean_value) / std_value
            df_test.loc[:, cols[j]] = (df_test.loc[:, cols[j]] - mean_value) / std_value
        else:
            raise NameError('This normalization is not included!')
    return df_test


def generate_sequences(df, win_len, diff_order, s_cols, c_cols):
    labels = []
    engine_id = []
    cycles = []
    data = []
    work_condition = []
    engine_number = df['ID'].max()
    for i in range(engine_number):
        index = df[df['ID'] == i + 1].index
        # whether the length of cycle smaller than win_len, if smaller, we pad the closet point in the front
        k = len(index)
        diff_df = df.loc[index, s_cols].values
        diff_df = diff_df[:, :, np.newaxis]
        for r in range(1, diff_order+1):
            diff_df = np.concatenate((diff_df, np.diff(np.concatenate((np.zeros((1, len(s_cols))), diff_df[:, :, r-1])), axis=0)[:,:,np.newaxis]), axis=2)
        if k < win_len:
            labels.append(df.loc[index[-1], 'RUL'])
            engine_id.append(i + 1)
            cycles.append(index[-1])
            data.append(np.concatenate((np.tile(diff_df[0, :, :], (win_len-k, 1, 1)), diff_df), axis=0))
            work_condition.append(np.vstack((np.ones((win_len - k, len(c_cols))) * df.loc[index[0], c_cols].values.reshape(1, -1),
                                  df.loc[index, c_cols].values.reshape(k, -1))))
        else:
            for j in range(k-win_len+1):
                labels.append(df.loc[index[win_len+j-1], 'RUL'])
                engine_id.append(i + 1)
                cycles.append(win_len+j)
                data.append(diff_df[j:j+win_len, :, :])
                work_condition.append(df.loc[index[j:j + win_len], c_cols].values)
    prepared_df = pd.DataFrame()
    prepared_df['engine_id'] = engine_id
    prepared_df['cycle'] = cycles
    prepared_df['data'] = data
    prepared_df['label'] = labels
    prepared_df['work_condition'] = work_condition
    return prepared_df


def generate_val_sequences(df, win_len, diff_order, s_cols, c_cols):
    labels = []
    engine_id = []
    cycles = []
    work_condition = []
    data = []
    engine_number = df['ID'].max()
    for i in range(engine_number):
        index = df[df['ID'] == i + 1].index
        # whether the length of cycle smaller than win_len, if smaller, we pad the closet point in the front
        k = len(index)
        diff_df = df.loc[index, s_cols].values
        diff_df = diff_df[:, :, np.newaxis]
        for r in range(1, diff_order + 1):
            diff_df = np.concatenate((diff_df, np.diff(np.concatenate((np.zeros((1, len(s_cols))), diff_df[:, :, r-1])), axis=0)[:,:,np.newaxis]), axis=2)
        if k < win_len:
            labels.append(df.loc[index[-1], 'RUL'])
            engine_id.append(i + 1)
            cycles.append(index[-1])
            data.append(np.concatenate((np.tile(diff_df[0, :, :], (win_len - k, 1, 1)), diff_df), axis=0))
            work_condition.append(np.vstack((np.ones((win_len-k, len(c_cols)))*df.loc[index[0], c_cols].values.reshape(1, -1),
                                  df.loc[index, c_cols].values.reshape(k, -1))))
        else:
            labels.append(df.loc[index[-1], 'RUL'])
            engine_id.append(i + 1)
            cycles.append(k)
            data.append(diff_df[k-win_len:, :, :])
            work_condition.append(df.loc[index[k-win_len:], c_cols].values)
    prepared_df = pd.DataFrame()
    prepared_df['engine_id'] = engine_id
    prepared_df['cycle'] = cycles
    prepared_df['data'] = data
    prepared_df['label'] = labels
    prepared_df['work_condition'] = work_condition
    return prepared_df


def add_rul(df, total_rul, end_point):
    engine_number = df['ID'].max()
    for i in range(engine_number):
        index = df[df['ID'] == i + 1].index
        # whether the true RUL larger than assumed total RUL
        if end_point[i] > total_rul:
            df.loc[index, 'RUL'] = total_rul
        else:
            rul = df.loc[index, 'Cycle'].sort_values(ascending=False) - 1 + end_point[i]
            rul[0: len(index) - total_rul + int(end_point[i])] = total_rul
            df.loc[index, 'RUL'] = list(rul)
    return df


def local_load_data(data_path):
    raw_data = pd.read_csv(data_path, delim_whitespace=True, header=None)
    new_cols = ['ID', 'Cycle', 'C1', 'C2', 'C3'] + ['S' + str(x) for x in range(1, 26 - 4)]
    raw_data.columns = new_cols

    # Drop the useless columns
    drop_list = ['S1', 'S5', 'S6', 'S10', 'S16', 'S18', 'S19']
    raw_data = raw_data.drop(drop_list, axis=1)
    return raw_data


if __name__ == "__main__":
    train_data_path = 'E:/GGCN for RUL prediction/CMAPSSData/train_FD001.txt'
    test_data_path = 'E:/GGCN for RUL prediction/CMAPSSData/test_FD001.txt'
    test_RUL_path = 'E:/GGCN for RUL prediction/CMAPSSData/RUL_FD001.txt'
    adj_path = 'E:/GGCN for RUL prediction/CMAPSSData/adj_001.txt'
    data_path = [train_data_path, test_data_path, test_RUL_path, adj_path]
    am = prepare_adj(data_path)
    datasets = {}
    datasets['train'] = CMAPSS(data_path=data_path, normalization_type='z-score',
                               win_len=30,  diff_order=3, train_mode='train')
    datasets['val'] = CMAPSS(data_path=data_path, normalization_type='z-score',
                             win_len=30, diff_order=3, train_mode='val')
    datasets['test'] = CMAPSS(data_path=data_path, normalization_type='z-score',
                              win_len=30,  diff_order=3, train_mode='test')
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=256,
                                                  shuffle=(True if x == 'train' else False),
                                                  num_workers=0,
                                                  pin_memory=False)
                   for x in ['train', 'val']}
    for batch_idx, (inputs, labels, work_condition, cycles) in enumerate(dataloaders['train']):
        print(inputs.size())
        print(labels.size())






