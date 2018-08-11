House Price Analysis:-

In this project, I have taken a Data set of House prices to see what are the parameters, 
which could be used predict the price accurately to help a business make informed 
Decisions.

So far,

- I have plotted Living area and Sale price. The plot indicates Living area has significant linear relation with Sale price.

- The overall quality of house is directly proportional with Sale price.

- The Total Basement area of a house has more of a quadratic relation between Sale price. After 2000 Square Feet Sale price increase is qudratic.

- For YearBuilt the Sale Price is linear initially but, it increases qudratically after the year 1960.

- The HeatMap Feature is used to find Correlation between various numerical attributes and, to decide whether to use them or to drop them. 

- I have compared dependent variable SalePrice for linearity asuumption. I also tried to plot SalePrice to compare its distribution with normal distribution where, I have realized the data is right skewed.
Therefore, I performed log transformation to convert SalePrice to a Normal distribution.

- I have selected highly correlated attributes with more than 0.5 correlation. I have also tried to draw a pairplot of highly correlated attributes.

- Attributes with more than 0.5 corraltion with SalePrice are taken off from the model.

-While imputing missing values, attributes with more than 50% of missing values are imputed with None.

- Lotfrontage missing values are imputed by neighbourhood lot frontage values.

- Categorical Garage mssing values are imputed with None, while numerical with 0.

- In the final step, I have selected variables with less than 0.5 correlation to build the final model.

=> Linear Regression model is giving an accuracy of 86.2665070745 %.

=> RandomForest Regression is giving an accurancy of 90.46438384 %.

=> GradientBoosting Regression is giving an accuracy of 92.1761126819 %. 