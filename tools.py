import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import os, sys

plt.ion()

def plot(mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Entrainement...')
    plt.xlabel('Nombre de générations')
    plt.ylabel('Moyenne de score (30 last scores)')
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def decreasing_exp(n,x) :
    return np.exp(-x/n)
