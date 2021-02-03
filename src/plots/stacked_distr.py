import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cols = ['School', 'C', 'D', 'I', 'M', 'O', 'S', 'Count', 'Percentage']
data = [['Oš Alojzija Šuštarja', 57, 94, 13, 14, 12, 0, 190, 0.053657159],
['OŠ Franca Rozmana Staneta', 346, 199, 71, 6, 11, 15, 648, 0.182999153],
['OŠ Ketteja in Murna', 206, 110, 97, 8, 100, 0, 521, 0.147133578],
['OŠ Koseze', 91, 205, 4, 8, 2, 14, 324, 0.091499576],
['OŠ Nove Fužine', 118, 89, 71, 47, 21, 1, 347, 0.097994917],
['OŠ Šmartno pod Šmarno goro', 77, 139, 15, 23, 15, 6, 275, 0.077661677],
['OŠ Valentina Vodnika', 123, 157, 60, 2, 42, 0, 384, 0.108443942],
['OŠ Vide Pregarc', 215, 56, 73, 11, 39, 1, 395, 0.111550409],
['OŠ Vižmarje - Brod', 197, 148, 13, 58, 40, 1, 457, 0.129059588]]


df = pd.DataFrame(data, columns=cols)

df['D'] = df['C'] + df['D']
df['I'] = df['D'] + df['I']
df['M'] = df['I'] + df['M']
df['O'] = df['M'] + df['O']
df['S'] = df['O'] + df['S']
df['Percentage'] *= 100

df = df.sort_values(by='Count', ascending=False)
df.reset_index(drop=True, inplace=True)
df.head
# DATA END

f, ax = plt.subplots(figsize=(10, 7))
sns.despine(f, left=True, bottom=True)

bg_color = 'white'
sns.set(rc={"font.style":"normal",
            "axes.facecolor":bg_color,
            "figure.facecolor":bg_color,
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':16,
            'figure.figsize':(10.0, 7.0),
            'xtick.labelsize':19,
            'font.size':16,
            'ytick.labelsize':14})

color_palette = sns.color_palette("tab10")
sns.barplot(data=df, y = 'S', x = 'School', color = color_palette[5], linewidth=0, dodge=False, label='Switching')
sns.barplot(data=df, y = 'O', x = 'School', color = color_palette[4], linewidth=0, dodge=False, label='Other')
sns.barplot(data=df, y = 'M', x = 'School', color = color_palette[3], linewidth=0, dodge=False, label='Moderation')
sns.barplot(data=df, y = 'I', x = 'School', color = color_palette[2], linewidth=0, dodge=False, label='Identity')
sns.barplot(data=df, y = 'D', x = 'School', color = color_palette[1], linewidth=0, dodge=False, label='Discussion')
sns.barplot(data=df, y = 'C', x = 'School', color = color_palette[0], linewidth=0, dodge=False, label='Chatting')


pos = range(len(df['Count']))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], df['Count'][tick] + 10, f"{df['Percentage'][tick]:.1f} %", horizontalalignment='center', fontsize='small')

ax.legend(ncol=1, loc="upper right", frameon=True, fontsize='medium')
#leg = p.get_legend()
#leg.set_title("")
#labs = leg.texts
#labs[0].set_fontsize(16)
#labs[1].set_fontsize(16)
#labs[2].set_fontsize(16)
ax.axes.xaxis.label.set_text("")
ax.axes.yaxis.label.set_text("")
#plt.savefig(os.path.join('../results', 'feature_importance.png'), format='png')

ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize='x-small')


plt.show()