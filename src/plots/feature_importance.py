

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



if os.getcwd().split('/')[-1] == 'plots':
    os.chdir('..')

# DATA START
from classifier_handcrafted_features.model import HandcraftedFeatures

inds = ['Book relevance', 'Type', 'CategoryBroad']
#cols = ['#tokens', '#mistakes in words', 'max len of a word', '#chars', '#?', '#!', '#,', '#.', '#caps',
#        '#interior caps', '#strange letters', '#interior numbers', 'lev. distance', '#names', '#quest_w', '#who']

cols = ['Token length', 'Mistakes count', 'Maximum word length', 'Character count', '\'?\' count', '\'!\' count',
        '\',\' count', '\'.\' count', 'Capital letter count', 'Inside capital letter count',
        'Strange letters count', 'Inside digits count', 'Levenshtein distance', 'Names count',
        'Quest words count', '\'Who\' count']


imp = []
for i, target in enumerate(inds):
    imp.append(HandcraftedFeatures('RF', target=target).feature_importances())

importance = pd.DataFrame(imp, columns=cols, index=inds).T
importance['feature'] = importance.index
importance.reset_index(drop=True, inplace=True)
importance = importance.sort_values('Book relevance', ascending=False)
importance = pd.melt(importance, id_vars=['feature'], value_vars=inds)

df = importance
# DATA END


df.head


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
                  y = 'feature',
                  x = 'value',
                  hue = 'variable',
                  palette="Blues",
                  saturation=2,
                  edgecolor=(0,0,0),
                  linewidth=0)

leg = p.get_legend()
leg.set_title("")
labs = leg.texts
labs[0].set_fontsize(16)
labs[1].set_fontsize(16)
labs[2].set_fontsize(16)
p.axes.xaxis.label.set_text("Importance")
p.axes.yaxis.label.set_text("Feature")
#plt.savefig(os.path.join('../results', 'feature_importance.png'), format='png')
plt.show()