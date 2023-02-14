# Car-price-prediction-final-model.
Car Price Prediction
1. Introduction
2. Loading Data and Explanation of Features
3. Exploratory Data Analysis (EDA)
4.	Feature Engineering
5.	Feature Selection
6. Applying Regression Models
7. Experimental Results
8. CONCLUSION


1.	Introduction :
Car price prediction is an important task in the automobile industry. Machine learning algorithms can be used to predict the price of a car based on various features such as make, model, year of manufacture, mileage, and more.
This project aims to solve the problem of predicting the price of a used car, using Sklearn's supervised machine learning techniques.
It is clearly a regression problem and predictions are carried out on dataset of used car sales. Several regression techniques have been studied, including Linear Regression, Decision Trees and Random forests of decision trees. Their performances were compared in order to determine which one works best with out dataset. 


2.	Loading Data and Explanation of Features:

The datasets consist of several independent variables include:
•	Name
•	Year
•	km_driven
•	fuel
•	seller_type
•	transmission
•	owner
•	mileage
•	engine
•	max_power
•	torque
•	seats

Dependent variable include
•	selling_price


3.	Exploratory Data Analysis (EDA):

Following steps are performed to do exploratory data analysis on the data

a.	looking describe dataset
b.	make dtypes of some variable 'category'
c.	create 'car_brand_name' feature from 'name' feature
d.	extract value of engine and mileage variable
e.	extract value of 'max_power' variable
f.	create 'car_age' feature from 'year' column
g.	drop the features of 'name','year' and 'torque'
h.	describe price value
i.	Avg price= 638271
ii.	Min price= 29999
iii.	Max price= 10000000 
i.	description of numeric variable
i.	minimum selling price is 29999 USD and maximum price is 10000000 USD and average selling price is 638271 USD.
ii.	The driving distance of the least driven car is 1 km, the most driven car's driving distance is 2360457 km, and average driving distance is 69819.
iii.	The no. of seats of cars changes from 2 seats to 14 seats
iv.	Minimum mileage is 0, maximum mileage is 42, average mileage is 19.4
v.	Engine volume changes from 624 to 3604, average is 1458.
