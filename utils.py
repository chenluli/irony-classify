import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import os

def draw_curves(arr, y_name="loss")
    color = cm.viridis(0.7)
    f, ax = plt.subplots(1,1)
    
    epoches = [i for i in range(len(arr))]
    ax.plot(epoches, arr, color=color)

    ax.set_xlabel('epoches')
    ax.set_ylabel('loss')

    plt.show()
    