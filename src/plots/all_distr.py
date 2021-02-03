


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cols = ['Type', 'Category', 'Count']
data = [['Book relevance', 'No', 2155],
['Book relevance', 'Yes', 1386],
['Type', 'Statement', 1711],
['Type', 'Answer', 1156],
['Type', 'Question', 674],
['CategoryBroad', 'Chatting', 1430],
['CategoryBroad', 'Discussion', 1197],
['CategoryBroad', 'Identity', 417],
['CategoryBroad', 'Other', 282],
['CategoryBroad', 'Moderation', 177],
['CategoryBroad', 'Switching', 38]]



df = pd.DataFrame(data, columns=cols)
#importance['feature'] = importance.index
#importance.reset_index(drop=True, inplace=True)
#importance = importance.sort_values('Book relevance', ascending=False)
#importance = pd.melt(importance, id_vars=['feature'], value_vars=inds)
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




p = sns.barplot(data=df,
                  y = 'Category',
                  x = 'Percentage',
                  hue = 'Type',
                  palette="Blues",
                  saturation=2,
                  edgecolor=(0,0,0),
                  linewidth=0,
                dodge=False)

pos = range(len(df['Count']))
for tick,label in zip(pos,p.get_yticklabels()):
    p.text(df['Percentage'][tick] + 0.01, pos[tick], df['Count'][tick], verticalalignment='center')


leg = p.get_legend()
leg.set_title("")
labs = leg.texts
labs[0].set_fontsize(16)
labs[1].set_fontsize(16)
labs[2].set_fontsize(16)
p.axes.xaxis.label.set_text("Percentage")
p.axes.yaxis.label.set_text("")
#plt.savefig(os.path.join('../results', 'feature_importance.png'), format='png')


plt.show()