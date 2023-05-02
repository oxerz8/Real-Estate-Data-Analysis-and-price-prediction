# Real Estate Data Analysis and price prediction
## Summary

Given a dataset of houses along with several attributes, we perform the following:
* Data cleaning using Pandas
* Data visualization using Seaborn
* Data manipulation using Numpy
* Application of machine learning models such as Linear Regression and PolynomialFeatures to determine the price of a house

### Conclusions
* Linear regression with Degree 2 PolynomialFeatures was an acceptable fit to the test data.

### Challenges
* The data contained Null/missing values that needed to be filled.
* Some of the datatypes were incorrect.
* Catagorical attributes needed to be encoded into a numeric form to fit the model
* Correlation was observed between some of the Quantitative columns

### Techniques Used
* One-hot encoding
* Linear Regression
* PolynomialFeatures

### Having more time
* The model could have been improved in the following ways:
  * Fine-tuning of hyperparameters
  * XGBoost could've been used to fit the data better through gradient boosting
  
## Source:
The data set was obtained from the Kaggle competition [House Prices - Advanced Regression Techniques: Predict sales prices and practice feature engineering, RFs, and gradient boosting](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

## Techniques:

### Acquisition:
The datasets were acquired in the form of an Excel file and were imported using the pandas function pd.read_csv(filename) into a dataframe

### Data preparation:

Given the following test data: 
| MSSubClass    | MSZoning      | LotFrontage   |...   |  SaleCondition | SalePrice |              
| ------------- | ------------- | ------------- |  --- | --- | --- |
| 60            | RL            | 65.0          |     ...   | Normal | 208500|                                           
| 20            | RL            | 80.0          |      ... | Normal |         181500|              

The data set contains 1460 records with 80 different features and the dependent variable to be predicted: SalePrice 

#### Data cleaning

![image](https://user-images.githubusercontent.com/23288977/235551355-80c035d4-4a8e-420c-a975-5bf1ab1eea12.png)

The distribution of the NULL values was checked for using seaborn heatmap.

Then a bar graph was plotted using matplotlib to check which features contained most of the missing values.

![image](https://user-images.githubusercontent.com/23288977/235551848-185f6e2f-d26c-4466-8d67-5fff3ac3b98e.png)

The features with over 80% null values were dropped using df.drop().

For the Catagorical features, the missing values were imputed with the mode of the attribute.

```
for column in Catagorical.columns:
    Xtrain[column].fillna(Xtrain[column].mode(), inplace=True)
```

For the Quantitative features, the missing values were imputed with the mean of the attribute.

```
for column in Quantitative.columns:
    Xtrain[column].fillna(Xtrain[column].mean(), inplace=True)
```

#### Data transformation

The catagorical features were transformed into numeric attributes using one-hot encoding.

```
Numerics = LabelEncoder()
Xtrain_n = Xtrain.copy()

Xtrain_n['MSZoning_n'] = Numerics.fit_transform(Xtrain_n['MSZoning'])
Xtrain_n['Street_n'] = Numerics.fit_transform(Xtrain_n['Street'])
```

### Data Analysis:

Correlation heatmap was plotted using seaborn for the quantitative features.

![image](https://user-images.githubusercontent.com/23288977/235556947-c2f901e8-8d89-4034-9bd4-5c15d7ba10bd.png)

The distribution of Saleprice was plotted using histogram from matplotlib

![image](https://user-images.githubusercontent.com/23288977/235557294-58c0f6f6-b344-433d-8198-b492ed769e00.png)
