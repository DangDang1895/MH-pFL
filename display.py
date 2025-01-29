import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import matplotlib  
import matplotlib  
matplotlib.use('svg')  
import matplotlib.pyplot as plt

class ExperimentLogger:
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]

def show_plot(client_stats,communication_rounds,dir):
    clear_output(wait=True)
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    acc_mean = np.mean(client_stats.client_acc, axis=1)
    acc_std = np.std(client_stats.client_acc, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Acc")
    plt.title("Client_acc")
    plt.xlim(0, communication_rounds)

    plt.subplot(1,2,2)
    acc_mean = np.mean(client_stats.client_loss, axis=1)
    acc_std = np.std(client_stats.client_loss, axis=1)
    plt.fill_between(client_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_stats.rounds, acc_mean, color="C0")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Loss")
    plt.title("Clients_loss")
    plt.xlim(0, communication_rounds)

    plt.savefig(dir)
    plt.close()






