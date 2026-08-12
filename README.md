# Learning Machine Learning Models and Algorithms (Python)
## About
While my knowledge of Machine Learning has a long journey ahead, this github repo intends to chart my journey as I learn various models and algorithms in order to deepen my understanding of machine learning using python. 

Therefore, the purpose of this project is not to present optimized or efficient code but rather to chart my learning process.

## AirBnb Price Predictor
Given AirBnB data from New York City, I created a price predictor that considers features such as amenities, location, and availability to predict housing prices. With this model, 65% of the change in prices can be associated to this models chosen features.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split

X = amenities_df.drop(columns=['price', 'revenue_per_yr'])
y = np.log1p(amenities_df['price']) #price is skewwed right

#train data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.2)

#Load model
model = TransformedTargetRegressor(
    regressor=RandomForestRegressor(),
    func=np.log1p, 
    inverse_func=np.expm1)
model.fit(Xtrain, ytrain)

# Calculate the R² score on test data
r2_accuracy = model.score(Xtest, ytest)
print(f"R² Score: {r2_accuracy}")
```
```text
R² Score: 0.6579160072669454
```

## Miscellaneous
### Data Cleaning
Cleaned AirBnb listings, State info, and Titatnic datasets in preparation for running Machine Learning Models. 
### Revenue Predictor
Created a revenue predictor albeit not very accurate. *I want to improve this in the future

## Works Cited
https://github.com/jakevdp/PythonDataScienceHandbook

https://insideairbnb.com/get-the-data/
