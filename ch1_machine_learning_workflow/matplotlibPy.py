import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("./ch1_machine_learning_workflow/mtcars-parquet.csv")

sns.set_theme(style="darkgrid")
sns.histplot(data=df, x="mpg")  # Pastikan Anda sudah mengimport dataframe
plt.show()
