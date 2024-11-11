import numpy as np
import glob
import matplotlib.pyplot as plt

def mapping_name(name):
    if "Adam" in name:
        return "Adam"
    if "AdGD" in name:
        return "AdGD"
    if "SNGDh" in name:
        return "SNGDh"
    if "SNGDn" in name:
        return "SNGDn"
    if "SGDm" in name:
        return "SGDm"
    if "SGD" in name:
        return "SGD"


def show_plot(dataset, metric):
    if metric == "losses":
        files = glob.glob(f"Result/{dataset}*/*_l.npy")
    if metric == "stepsize":
        files = glob.glob(f"Result/{dataset}*/*_lr.npy")
    values = []
    labels = []
    for file in files:
        labels.append(mapping_name(file))
        values.append(np.load(file))

    n_plot = 400
    plt.figure(figsize=(16, 12))
    for i, val in enumerate(values):
        plt.plot(
            np.arange(0, len(val)),
            val, label=labels[i])
    plt.yscale('log')
    plt.xlabel('Epoch') # 'Epoch' / 'Iteration'
    plt.ylabel(r'Trainloss') # 'Train loss' / 'Stepsize'
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{dataset}_{metric}.pdf', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    dataset = "mnist" # FashionMNIST/cifar10/mnist
    metric = "losses" # losses/stepsize
    show_plot(dataset, metric)