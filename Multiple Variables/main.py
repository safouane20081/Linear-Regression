import pandas as pd 
import numpy as np
from sklearn import linear_model
file=pd.read_csv("manhattan_area_rooms_age_price.csv")
model=linear_model.LinearRegression()
model.fit(file[["area","rooms","age"]],file["price"])
print(model.predict([[300,6,5]])[0])