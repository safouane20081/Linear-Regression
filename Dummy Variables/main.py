import pandas as pd 
import numpy as np
from sklearn import linear_model
file=pd.read_csv("manhattan_town_area_price.csv")
dummy=pd.get_dummies(file["town"])
merged = pd.concat([file, dummy], axis="columns")
update=merged.drop(["town","area"],axis="columns")
print(update)
x=update.drop(["price"],axis="columns")
y=update["price"]
model=linear_model.LinearRegression()
model.fit(x,y)
# List of all towns
towns = list(dummy.columns)

# Ask user for input
area = float(input("Enter area in sqft: "))
town_name = input("Enter town name: ")

# Create a list of zeros
town_values = []
for town in towns:
    if town == town_name:
        town_values.append(1)
    else:
        town_values.append(0)

# Create the input dataframe
input_data = pd.DataFrame([town_values], columns=towns)

# Predict
result = model.predict(input_data)
print(f"Predicted price: ${result[0]:,.2f}")