import pandas as pd
import plotly.express as px


df = pd.read_csv("Combined Data.csv")
df.info()
df.describe()
print(f"Der Datensatz hat {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")
df.head()
df.columns

#Drop empty column
df.drop(columns=["Unnamed: 0"], inplace=True)

#Histogram of the mental health status
fig = px.histogram(df, x='status', title='Distribution of Mental Health Status')
fig.show()

#Show the first 5 statements of the dataset in lowercase
df.statement = df.statement.str.lower()
df.head()
