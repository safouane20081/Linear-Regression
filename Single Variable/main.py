import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
file=pd.read_csv("manhattan_area_price.csv")
plt.xlabel("area (square metre)")
plt.ylabel("price (US dollar)")
plt.scatter(file["area"].head(10),file["price"].head(10),color="green",marker="*")
plt.show()
model=linear_model.LinearRegression()
model.fit(file[["area"]],file["price"])
print(model.predict([[300]])[0])

