import os
import random
import torch
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# We simply chose the smallest data type which fits the data in columns
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def read_data(input_dir):
    sell_prices_df = pd.read_csv(input_dir + 'sell_prices.csv')
    sell_prices_df = reduce_mem_usage(sell_prices_df)
    print('Sell prices has {} rows and {} columns'.format(sell_prices_df.shape[0], sell_prices_df.shape[1]))

    calendar_df = pd.read_csv(input_dir + 'calendar.csv')
    calendar_df = reduce_mem_usage(calendar_df)
    print('Calendar has {} rows and {} columns'.format(calendar_df.shape[0], calendar_df.shape[1]))

    sales_train_validation_df = pd.read_csv(input_dir + 'sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))

    submission_df = pd.read_csv(input_dir + 'sample_submission.csv')
    return sell_prices_df, calendar_df, sales_train_validation_df, submission_df


def create_sales_df(calendar_df, sales_train_validation_df):
    # Create date index
    date_index = calendar_df['date']
    dates = date_index[0:1913]
    dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
    # Create a data frame for items sales per day with item ids (with Store Id) as columns names  and dates as the index
    sales_train_validation_df['item_store_id'] = sales_train_validation_df.apply(
        lambda x: x['item_id'] + '_' + x['store_id'], axis=1)
    DF_Sales = sales_train_validation_df.loc[:, 'd_1':'d_1913'].T
    DF_Sales.columns = sales_train_validation_df['item_store_id'].values

    # Set Dates as index
    DF_Sales = DF_Sales.set_index([dates_list])
    DF_Sales.index = pd.to_datetime(DF_Sales.index)
    return DF_Sales

# extract sales data with the given category in the given store
def data_subset(sales_df, category, store):
    columns = [c for c in sales_df.columns if category in c and store in c]
    return sales_df[columns]

def train_subset(data, num_days_included=365):
    all_data = np.array(data)
    # Now we take last yer (365 days)
    data_subset = all_data[-num_days_included:, :]
    return data_subset


def scaled_data(data, scaler, plot=False):
    train_data_normalized = scaler.fit_transform(data)

    if plot:
        # we check that data distribution did not change after normalization
        fig, axs = plt.subplots(2)
        fig.suptitle('Data Distribution Before and After Normalization ', fontsize=19)
        pd.DataFrame(data).plot(kind='hist', ax=axs[0], alpha=.4, figsize=[12, 6], legend=False,
                                title=' Before Normalization', color='red')
        pd.DataFrame(train_data_normalized).plot(kind='hist', ax=axs[1], figsize=[12, 6], alpha=.4, legend=False,
                                                 title=' After Normalization' \
                                                 , color='blue')

    return train_data_normalized

###  This function creates a sliding window or sequences of 28 days and one day label ####
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# set validation_size > 0 only for model tuning and not for submission prediction
def train_test_split(dataset, validation_size=0, window=28):
    #use last 28 days as validation
    if validation_size > 0:
        train_size = len(dataset[:,0]) - (validation_size + window)

        train = dataset[:train_size]
        val = dataset[train_size:]
        return train, val
    else:
        return dataset, None


def plot_random_prediction(df_predict, df_labels=None):
    # Plot
    figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
    index = random.randint(0, len(df_predict)-1)
    plt.plot(df_predict[index])
    if df_labels is not None:
        plt.plot(df_labels[5])
        plt.legend(['Prediction','Time Series'],fontsize = 21)
    else:
        plt.legend(['Prediction'],fontsize = 21)
    plt.suptitle('Time-Series Prediction Test Set',fontsize = 23)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.ylabel(ylabel='Sales Demand',fontsize = 21)
    plt.xlabel(xlabel='Date',fontsize = 21)
    plt.show()


def create_submission(df_validation, df_evaluation, df_sales_grouped, category, store):
    # validation csv
    columns = [c + "_validation" for c in df_sales_grouped.columns]
    index = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14",
             "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28"]
    df_validation.columns = columns
    df_validation = df_validation.set_index([index])
    dfv_transposed = df_validation.T
    dfv_transposed.to_csv(r'{}_{}_validation.csv'.format(category, store))

    # evaluation csv
    columns = [c + "_evaluation" for c in df_sales_grouped.columns]
    index = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14",
             "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28"]
    df_evaluation.columns = columns
    df_evaluation = df_evaluation.set_index([index])
    dfe_transposed = df_evaluation.T
    dfe_transposed.to_csv(r'{}_{}_evaluation.csv'.format(category, store))


def main():
    # Demonstration of data generation
    # define input sequence
    in_seq1 = np.array([x for x in range(0, 100, 10)])
    in_seq2 = np.array([x for x in range(5, 105, 10)])
    in_seq3 = np.array([x for x in range(10, 110, 10)])
    out_seq = np.array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    in_seq3 = in_seq3.reshape((len(in_seq3), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # horizontally stack columns
    dataset = np.hstack((in_seq1, in_seq2, in_seq3, out_seq))
    print(dataset)

    t, v = train_test_split(dataset, 4, window=3)
    print(t)
    print(v)



if __name__ == '__main__':
    main()