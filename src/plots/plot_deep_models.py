import pickle

import matplotlib.pyplot as plt


def plot():
    results = pickle.load(open('../results/results_deep_models', 'rb'))

    results.plot.barh(title="Out of bag evaluation of the models", fontsize=14)
    plt.xlabel('F1', fontsize=14)
    plt.legend(loc=8)
    plt.tight_layout()
    plt.savefig('../results/plot_deep_models.pdf', format='pdf')


if __name__ == '__main__':
    plot()
