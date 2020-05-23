import pickle

import matplotlib.pyplot as plt


def plot():
    results = pickle.load(open('../results/deep_models', 'rb'))

    results.plot.barh(title="Out of bag evaluation of the models", fontsize=14)
    plt.xlabel('Accuracy', fontsize=14)
    plt.legend(loc=8)
    plt.tight_layout()
    plt.savefig('../results/hold_out_eval.pdf', format='pdf')


if __name__ == '__main__':
    plot()
