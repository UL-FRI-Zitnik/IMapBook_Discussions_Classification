import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_cf(df, ax_loc, font_scale=1.4, font_size=16):
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=font_scale)  # for label size
    res = sn.heatmap(df, annot=True, annot_kws={"size": font_size}, cmap="Blues", fmt="d", ax=ax_loc)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize=font_size)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=font_size)
    res.set_xlabel('Predicted label', fontsize=font_size)
    res.set_ylabel('True label', fontsize=font_size)


# Category - BERT Slovene
header = ['CB', 'CC', 'CE', 'CF', 'CG', 'CO', 'DA', 'DAA', 'DE', 'DQ', 'IA', 'IQ', 'IQA', 'MA', 'ME', 'MQ', 'O', 'S']
array = [[2, 0, 0, 0, 0, 3, 2, 7, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 5, 0, 0, 0, 3, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 8, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
[0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0],
[0, 2, 0, 0, 3, 152, 14, 12, 0, 8, 8, 8, 0, 0, 2, 2, 13, 0],
[1, 2, 0, 0, 1, 12, 120, 8, 0, 3, 2, 3, 0, 0, 3, 3, 2, 1],
[0, 0, 0, 0, 0, 11, 9, 18, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
[2, 0, 0, 0, 0, 6, 3, 2, 0, 15, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 1, 0, 0, 0, 13, 4, 0, 0, 0, 11, 0, 0, 0, 1, 0, 1, 0],
[0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 36, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 11, 2, 0, 0],
[0, 0, 0, 0, 0, 3, 5, 1, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0],
[1, 0, 0, 0, 0, 5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 33, 0],
[0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]

df = pd.DataFrame(array, header, header)
new_order = ['CO', 'DA', 'IQ', 'O', 'DAA', 'DQ', 'IA', 'ME','MQ','CG','CC','CB', 'MA','CE', 'CF',  'DE',   'IQA',    'S']
df = df.reindex(new_order, columns=new_order)
plot_cf(df, None)
#plt.savefig('../../results/category_conf_matrix.png')
plt.show()

# Category broad - BERT Slovene
header = ['C', 'D', 'I', 'M', 'O', 'S']
array = [[202,  41,  20,   6,  12,   1],
 [ 46, 169,  1,12,  2,  3],
 [ 18,   4, 48, 3,  1,  0],
 [  6,   6,  1,27,  0,  1],
 [  9,   1,  1,0, 30,  0],
 [  4,   0,  1,1,  0,  0]]

df = pd.DataFrame(array, header, header)
plot_cf(df, None)
#plt.savefig('../../results/broad_category_conf_matrix.png', bbox_inches = 'tight', pad_inches = 0)
plt.show()


fig, axs = plt.subplots(ncols=2, figsize=(15,7))

#Book relevance - BERT Slovene
header = ['No', 'Yes']
array = [[352,  45],
 [ 71, 209]]

df = pd.DataFrame(array, header, header)
plot_cf(df, axs[0], font_size=25, font_scale=2)


#Type - BERT Slovene
header = ['A', 'Q', 'S']
array = [[153,  10,  60],
 [  9, 112,  18],
 [ 39,  16, 260]]

df = pd.DataFrame(array, header, header)
new_order = ['S', 'A', 'Q']
df = df.reindex(new_order, columns=new_order)
plot_cf(df, axs[1], font_size=25, font_scale=2)
#fig.savefig('../../results/relevanceAndType_conf_matrix.png', bbox_inches = 'tight', pad_inches = 0)
fig.show()



