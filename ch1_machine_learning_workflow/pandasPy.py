import pandas as pd

df = pd.read_csv("./ch1_machine_learning_workflow/mtcars-parquet.csv")
print(df.head())
print(df.shape)
print(df["mpg"].mean())  # Output: 20.090625
print(
    df["model"].head()
)  # Output: Mazda RX4, Mazda RX4 Wag, Datsun 710, Hornet 4 Drive, Hornet Sportabout
