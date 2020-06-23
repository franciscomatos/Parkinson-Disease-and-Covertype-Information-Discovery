import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import scipy.stats as _stats
from sklearn.preprocessing import OneHotEncoder


# register_matplotlib_converters()
# data = pd.read_csv('data/algae.csv', index_col='date', parse_dates=True, infer_datetime_format=True)
# plt.figure(figsize=(5,4))
# plt.plot(data['pH'])
# plt.show()

# plt.figure(figsize=(12,4))
# plt.ylim(0, 14)
# plt.title('pH along time')
# plt.xlabel('date')
# plt.ylabel('pH')
# plt.plot(data['pH'])
# plt.show()

def choose_grid(nr):
    return nr // 4 + 1, 4

def line_chart(ax: plt.Axes, series: pd.Series, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.plot(series)

# (rows, cols) = choose_grid(data.shape[1])
# plt.figure()
# fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
# i, j, n = 0, 0, 0

# for col in data:
#     line_chart(axs[i, j], data[col], col, 'date', col)
#     n = n + 1
#     i, j = (i + 1, 0) if n % cols == 0 else (i, j + 1)
# fig.tight_layout()
# plt.show()

def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox = True, shadow = True)

# plt.figure(figsize=(12,4))
# two_series = {'Phosphate': data['Phosphate'], 'Orthophosphate': data['Orthophosphate']}
# multiple_line_chart(plt.gca(), data.index, two_series, 'Phosphate and Orthophosphate values', 'date', '')
# plt.show()


def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=90, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.bar(xvalues, yvalues, edgecolor='grey')

# plt.figure()    
# counts = data['season'].value_counts()
# bar_chart(plt.gca(), counts.index, counts.values, 'season distribution', 'season', 'frequency')
# plt.show()


def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    k = 0
    for name, y in yvalues.items():
        ax.bar(x + k * step, y, step, label=name)
        k += 1
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox = True, shadow = True)


# two_series = {'river_depth': data['river_depth'].value_counts().sort_index(), 
#               'fluid_velocity': data['fluid_velocity'].value_counts().sort_index()}
# plt.figure()
# multiple_bar_chart(plt.gca(), ['high', 'low', 'medium'], two_series, '', '', 'frequency')
# plt.show()

def compute_known_distributions(x_values, n_bins) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # LogNorm
  #  sigma, loc, scale = _stats.lognorm.fit(x_values)
  #  distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
   # a, loc, scale = _stats.skewnorm.fit(x_values)
   # distributions['SkewNorm(%.2f)'%a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    return distributions

def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    n, bins, patches = ax.hist(values, 20, density=True, edgecolor='grey')
    distributions = compute_known_distributions(values, bins)
    multiple_line_chart(ax, values, distributions, 'Best fit for %s'%var, var, 'probability')


def dummify(df, cols_to_dummify):
    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')

    for var in cols_to_dummify:
        one_hot_encoder.fit(df[var].values.reshape(-1, 1))
        feature_names = one_hot_encoder.get_feature_names([var])
        transformed_data = one_hot_encoder.transform(df[var].values.reshape(-1, 1))
        df = pd.concat((df, pd.DataFrame(transformed_data, columns=feature_names)), 1)
        df.pop(var)

    return df