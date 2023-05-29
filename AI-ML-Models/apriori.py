import numpy as np  # linear algebra
import pandas as pd  # data processing
import plotly.express as px
import apyori
from apyori import apriori

data = pd.read_csv("Groceries_dataset.csv")
data.head()

print("Top 10 frequently sold products(Tabular Representation)")
x = data['itemDescription'].value_counts().sort_values(ascending=False)[:10]
transactions = x
transactions[0:10]
fig = px.bar(x=x.index, y=x.values)
fig.update_layout(title_text="Top 10 frequently sold products (Graphical Representation)",
                  xaxis_title="Products", yaxis_title="Count")
fig.show()


data["Year"] = data['Date'].str.split("-").str[-1]
data["Month-Year"] = data['Date'].str.split(
    "-").str[1] + "-" + data['Date'].str.split("-").str[-1]
fig1 = px.bar(data["Month-Year"].value_counts(ascending=False),
              orientation="v",
              color=data["Month-Year"].value_counts(ascending=False),
              labels={'value': 'Count', 'index': 'Date', 'color': 'Meter'})

fig1.update_layout(title_text="Exploring higher sales by the date")

fig1.show()

rules = apriori(transactions, min_support=0.00030,
                min_confidence=0.05, min_lift=3, max_length=2, target="rules")
association_results = list(rules)
print(association_results[0])

for item in association_results:

    pair = item[0]
    items = [x for x in pair]

    print("Rule : ", items[0], " -> " + items[1])
    print("Support : ", str(item[1]))
    print("Confidence : ", str(item[2][0][2]))
    print("Lift : ", str(item[2][0][3]))

    print("=============================")
