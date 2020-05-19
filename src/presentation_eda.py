import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# Read in the data
INPUT_DIR = 'data'
cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
stv = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
sellp = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')

d_cols = [c for c in stv.columns if 'd_' in c] # sales data columns

cal[['d','date','event_name_1','event_name_2',
     'event_type_1','event_type_2', 'snap_CA']].head()

past_sales = stv.set_index('id')[d_cols] \
    .T \
    .merge(cal.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')


def plot_product_categories():
    categories = ['FOODS', 'HOUSEHOLD', 'HOBBIES']
    titles = ['Foods', 'Household', 'Hobbies']
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    handles, labels = [], []
    for i in [0, 1, 2]:
        ax = axes[i]
        ax.set_title(titles[i])
        ax.set_ylim([-1000, 22500])
        ax.label_outer()
        cat1 = categories[i] + "_" + str(1)
        items1_col = [c for c in past_sales.columns if cat1 in c]
        axx = past_sales[items1_col] \
            .sum(axis=1) \
            .rolling(28).mean() \
            .plot(kind='line',
                  lw=3,
                  color=next(color_cycle),
                  ax=ax, rot=45, label=cat1)
        axx.set_xlabel("")

        cat2 = categories[i] + "_" + str(2)
        items2_col = [c for c in past_sales.columns if cat2 in c]
        axx = past_sales[items2_col] \
            .sum(axis=1) \
            .rolling(28).mean() \
            .plot(kind='line',
                  lw=3,
                  color=next(color_cycle),
                  ax=ax, rot=45, label=cat2)
        axx.set_xlabel("")

        if i == 0:
            cat3 = categories[i] + "_" + str(3)
            items3_col = [c for c in past_sales.columns if cat3 in c]
            axx = past_sales[items3_col] \
                .sum(axis=1) \
                .rolling(28).mean() \
                .plot(kind='line',
                      lw=3,
                      color=next(color_cycle),
                      ax=ax, rot=45, label=cat3)
            axx.set_xlabel("")

        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    fig.suptitle('Sales Trends by Product Category',
                 size=20,
                 y=1.1)

    plt.tight_layout()
    fig.legend(handles, labels, loc='center right')
    plt.subplots_adjust(right=0.88)
    plt.show()


def plot_stores():
    categories = ['CA', 'TX', 'WI']
    titles = ['California', 'Texas', 'Wisconsin']
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    handles, labels = [], []
    for i in [0, 1, 2]:
        ax = axes[i]
        ax.set_title(titles[i])
        ax.set_title(titles[i])
        ax.set_ylim([1000, 7500])
        ax.label_outer()
        cat1 = categories[i] + "_" + str(1)
        items1_col = [c for c in past_sales.columns if cat1 in c]
        axx = past_sales[items1_col] \
            .sum(axis=1) \
            .rolling(28).mean() \
            .plot(kind='line',
                  lw=3,
                  color=next(color_cycle),
                  ax=ax, rot=45, label=cat1)
        axx.set_xlabel("")

        cat2 = categories[i] + "_" + str(2)
        items2_col = [c for c in past_sales.columns if cat2 in c]
        axx = past_sales[items2_col] \
            .sum(axis=1) \
            .rolling(28).mean() \
            .plot(kind='line',
                  lw=3,
                  color=next(color_cycle),
                  ax=ax, rot=45, label=cat2)
        axx.set_xlabel("")

        cat3 = categories[i] + "_" + str(3)
        items3_col = [c for c in past_sales.columns if cat3 in c]
        axx = past_sales[items3_col] \
            .sum(axis=1) \
            .rolling(28).mean() \
            .plot(kind='line',
                  lw=3,
                  color=next(color_cycle),
                  ax=ax, rot=45, label=cat3)
        axx.set_xlabel("")

        if i == 0:
            cat4 = categories[i] + "_" + str(4)
            items4_col = [c for c in past_sales.columns if cat4 in c]
            axx = past_sales[items4_col] \
                .sum(axis=1) \
                .rolling(28).mean() \
                .plot(kind='line',
                      lw=3,
                      color=next(color_cycle),
                      ax=ax, rot=45, label=cat4)
            axx.set_xlabel("")

        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    fig.suptitle('Sales Trends by Store',
                 size=20,
                 y=1.1)

    plt.tight_layout()
    fig.legend(handles, labels, loc='center right')
    plt.subplots_adjust(right=0.92)
    plt.show()


plot_product_categories()