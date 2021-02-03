


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cols = ['Category', 'Count']
data = [['CB', 96],
['CC', 77],
['CE', 30],
['CF', 19],
['CG', 97],
['CO', 1110],
['DA', 833],
['DAA', 210],
['DE', 32],
['DQ', 122],
['IA', 191],
['IQ', 221],
['IQA', 6],
['MA', 29],
['ME', 69],
['MQ', 79],
['O', 282],
['S', 38]]

df = pd.DataFrame(data, columns=cols)
df = df.sort_values(by='Count', ascending=False)
df.reset_index(drop=True, inplace=True)
df['Percentage'] = df['Count'] / max(df['Count'])
df.head
# DATA END


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
            'xtick.labelsize':12,
            'font.size':12,
            'ytick.labelsize':12})



color_palette = sns.color_palette("Blues")

p = sns.barplot(data=df,
                  y = 'Category',
                  x = 'Percentage',
                  color = color_palette[2],
                  saturation=2,
                  edgecolor=(0,0,0),
                  linewidth=0,
                dodge=False)

pos = range(len(df['Count']))
for tick,label in zip(pos,p.get_yticklabels()):
    p.text(df['Percentage'][tick] + 0.01, pos[tick], df['Count'][tick], verticalalignment='center')


#leg = p.get_legend()
#leg.set_title("")
#labs = leg.texts
#labs[0].set_fontsize(16)
#labs[1].set_fontsize(16)
#labs[2].set_fontsize(16)
p.axes.xaxis.label.set_text("Percentage")
p.axes.yaxis.label.set_text("")
#plt.savefig(os.path.join('../results', 'feature_importance.png'), format='png')


plt.show()