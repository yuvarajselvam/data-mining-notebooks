CSE 578: Data Visualization
Course Project Code
Yuvaraj Selvam
Computer Science and Engineering Department
Arizona State University, Tempe Arizona USA
yselvam@asu.edu

Appendix

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('Data/adult.data', index_col=False, names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"])
data.head(5)

data.describe()

# %%
data.info()

# %%
# data.drop(["capital-loss", "fnlwgt"], axis=1, inplace=True)

# %%
data[data.select_dtypes('object').columns] = data.select_dtypes('object').apply(lambda x: x.str.strip())
data.replace(r'^\?$', np.NaN, regex=True, inplace=True)

# %%
null_cols = [col for col in data.columns if any(data[col].isna())]
if null_cols:
    fillers = data[null_cols].mode().iloc[0].to_dict()
    data.fillna(fillers, inplace=True)
    print(f'Found {len(null_cols)} columns with null values.' + 
    f' Filling them with their respective modes:\n{fillers}')

# %%
data['agegroup'] = data['age'].apply(lambda x: f'{(x//10)*10}-{(x//10)*10 + 9}')
data[['age', 'agegroup']]

# %%
sns.set(style="darkgrid")
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots(figsize=(10, 12))
data[["agegroup", "income"]].groupby(['agegroup', 'income']).size().reset_index().pivot('agegroup', 'income')[0].plot(kind='barh', stacked=True, ax=ax)
ax.set_title('Count by income category for each age group\n')
ax.set_ylabel('agegroup\n')
ax.set_xlabel('\ncount')

# %%
corr = data.select_dtypes('int64').corr()
fig, ax = plt.subplots(1, 1, figsize=(12,12))
sns.heatmap(corr, ax=ax, square=True, linewidth=0.1, annot=True, annot_kws={'size': 10})
plt.title('Correlation heatmap\n', fontdict={'size': 18})

# %%
sns.pairplot(data[['income', 'education-num', 'hours-per-week']], hue='income')
fig = plt.gcf()
fig.set_figheight(7)
fig.set_figwidth(11)

# %%
ax = sns.relplot(x='education-num', y='hours-per-week', data=data, kind='scatter', hue='income', col='sex', col_wrap=1)
ax.fig.subplots_adjust(top=0.9)
ax.fig.suptitle('Occupation versus Hours worked per week split by sex')
fig = plt.gcf()
fig.set_figheight(5)
fig.set_figwidth(9)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=True)

# %%
sns.lineplot(x="education-num", y="capital-gain", data=data, hue='income')
plt.title('Line chart showing the trends in capital-gain versus education \nfor the two income categories\n')

# %%
sns.jointplot(
    x='fnlwgt', 
    y='capital-loss', 
    data=data,  
    hue='income')

# %%
g = sns.catplot(y='hours-per-week', x='occupation', data=data.sort_values(by="education-num"), 
                height=3, kind='box', aspect=4.5, hue='income', orient='v')
fig = plt.gcf()
fig.set_figheight(9)
# fig.set_figwidth(15)
g.set_xticklabels(rotation=90, ha='right')
plt.title('Box and Whisker plot of hours-per-week for different occupations\n\n')

# %%
sns.kdeplot(
    x='fnlwgt', 
    y='education-num',
    data=data,  
    hue='income', fill=True)

# %%
col = 'marital-status'

fig, ax = plt.subplots(figsize=(14, 14), nrows=2, ncols=1)

pie1 = data[data["income"] == '<=50K'][col].sort_values().value_counts().to_dict()
pie2 = data[data["income"] == '>50K'][col].sort_values().value_counts().to_dict()

props = {
    'autopct': lambda x: f'{round(x, 1)}%' if round(x,1) > 0.1 else '', 
    'wedgeprops': {'edgecolor': 'white', 'linewidth': 3}, 
    'pctdistance': 1.1,
}

wedges, labels, autopct = ax[0].pie(pie1.values(), **props)
[x.set_fontsize(9.2) for x in autopct]
wedges, labels, autopct = ax[1].pie(pie2.values(), **props)
[x.set_fontsize(9.2) for x in autopct]

ax[1].legend(pie1.keys(), loc='center', bbox_to_anchor=(.5, 1.1))

ax[0].text(0, 0, 'income <= 50K', horizontalalignment='center')
ax[1].text(0, 0, 'income > 50K', horizontalalignment='center')

ax[0].add_artist(plt.Circle((0,0),0.7,fc='white'))
ax[1].add_artist(plt.Circle((0,0),0.7,fc='white'))
fig.subplots_adjust(top=0.95)
fig.suptitle('Pie chart showing proportions of people in various marital statuses for each income category\n\n')


