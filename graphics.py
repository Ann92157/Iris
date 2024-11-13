import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data():
    snsdata = pd.read_csv(r"Iris.csv").drop(['Id'], axis=1)
    g = sns.pairplot(snsdata, hue='Species', markers='x')
    g = g.map_upper(plt.scatter)
    g = g.map_lower(sns.kdeplot)
    plt.savefig(f"data.png")
    plt.show()
