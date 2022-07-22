# Airline Passenger Satisfaction Analysis
**Analysis & Machine Learning of Airline Passenger Satisfaction Database**

*Associated with Purwadhika Coding School for Final Project*

<br />

Business Problem Understanding
------------------------------
**Problem Statement :**

What factors that affect customer satisfaction in airline industry?

Most companies realize that providing their customers with the best possible experience is a strategic need, but most state that they cannot manage it effectively. This problem is exacerbated by the large number of customer interaction points in the airline industry.

By utilizing data analysis and machine learning to predict customer satisfaction and determine which factors related strongly to customer satisfaction, we should be able to take countermeasures to prever dissatisfaction.

**Goals :**

In this project, we will explore from airline's point of view and to answer questions below:
1. What are the insights that can be drawn from the data we have?
2. What factors influence customer ratings significantly?
3. Is there any development that can be done to increase the level of customer satisfaction?

**Analytic Approach :**

To answer the questions above, we will analyze the data to find out what things that have impact on the assessment of customer satisfaction, followed by building a **classification** model that will help the company to see a comparison of satisfied and dissatisfied customers and what steps need to be taken to fix it.

**Metric Evaluation :**

Based on the consequences, we will create a model that can minimize 2 things:
1. The number of customers who are considered satisfied but actually are not *(False Positive)*. The possibility of them not coming back can harm the company revenue and prevent new customers to come (by word of mouth).
2. The number of incentives given to customers *(False Negative)* so that the outgoing costs are more efficient.

Although it appears that the consequences of False Positives are greater, we must also pay attention to the consequences of False Negatives. Therefore, we will use **f1-score** as measurements.

<br />

Data Understanding
------------------
Dataset source: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

This dataset is an airline customer satisfaction survey compiled by **John D in 2018** (https://www.kaggle.com/datasets/johndddddd/customer-satisfaction), which was later modified by **TJ Client in 2020** for classification model purposes. Based on the **timeframe**, this data is valid to be used in helping to solve existing problems.

Each row of the dataset represents a customer's flight data and the satisfaction they feel with the available services. (for details, please see the next section)

### Attribute Information

| Attribute | Data Type, Length | Description |
| --- | --- | --- |
| Gender | Text | Gender of the passengers (Female, Male) |
| Customer Type | Text | The customer type (Loyal customer, disloyal customer) |
| Age | Int | The actual age of the passengers |
| Type of Travel | Text | Purpose of the flight of the passengers (Personal Travel, Business Travel) |
| Class | Text | Travel class in the plane of the passengers (Business, Eco, Eco Plus) |
| Flight distance | Int |The flight distance of this journey |
| Inflight wifi service | Int | Satisfaction level of the inflight wifi service (0:Not Applicable;1-5) |
| Departure/Arrival time convenient | Int | Satisfaction level of Departure/Arrival time convenient |
| Ease of Online booking | Int | Satisfaction level of online booking |
| Gate location | Int | Satisfaction level of Gate location |
| Food and drink | Int | Satisfaction level of Food and drink |
| Online boarding | Int | Satisfaction level of online boarding |
| Seat comfort | Int | Satisfaction level of Seat comfort |
| Inflight entertainment | Int | Satisfaction level of inflight entertainment |
| On-board service | Int | Satisfaction level of On-board service |
| Leg room service | Int | Satisfaction level of Leg room service |
| Baggage handling | Int | Satisfaction level of baggage handling |
| Check-in service | Int | Satisfaction level of Check-in service |
| Inflight service | Int | Satisfaction level of inflight service |
| Cleanliness | Int | Satisfaction level of Cleanliness |
| Departure Delay in Minutes | Int | Minutes delayed when departure |
| Arrival Delay in Minutes | Float | Minutes delayed when Arrival |
| satisfaction | Int | Airline satisfaction level(Satisfied, neutral or dissatisfied) |

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Data%20Condition.PNG)

<br />

Data Analysis
-------------
![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Satisfaction%20by%20Class.PNG)
![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Class%20by%20Flight%20Distance.PNG)
![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Customer%20Type%20by%20Age%20Group.PNG)
![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Class%20by%20Type%20of%20Travel.PNG)

**Analysis :**

- Data is balanceed, where the number of customers who are not `not satisfied/neutral [0]` is 57% and those who are `satisfied [1]` are 43%
- Based on the `Class` feature:
  - More customers take Business and Eco classes compared to Eco Plus.
  - The number of satisfied customers is almost inversely proportional between the two classes, where there are more satisfied customers in the Business class.
  - Customers in the age group below 30 and above 60 mostly take the Eco class, while the age group 40 to 49 years mostly take the Business class.
- Based on the `Flight Distance` feature, customers with long flights tend to take Business class flights.
- Based on the `Customer Type` feature:
  - The number of loyal customers (82%) is higher than those who are not (18%)
  - The age of loyal customers is relatively older than those who are not.
- Based on the `Type of Travel` feature, customers who take Business class flights tend to have business-related purposes.

*Note: The analysis shown is only closely related to the prediction model and recommendations. For complete tables and analysis, please check the notebook file*

<br />

Modelling & Evaluation
----------------------
After doing cross validation, the best model to use is `CatBoost` with f1-score 0.956894 before tuning, and 0.957916 after tuning. Both scores were obtained from the training set. Scores had a slight improvement on test set. The comparison can be seen in the following figure:

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Score%20Comparison.PNG)

**Feature Importances :**

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Feature%20Importances.PNG)

It can be seen that for our CatBoost model, the `inflight wifi service` feature/column is the most important, followed by `Type of Travel`, `Customer Type`, and so on. We will use this graph as a reference in providing recommendations in the next section.

**SHAP Values :**

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/SHAP.PNG)

Based on **SHAP Values**, we can conclude:
1. `Class`: Business Class has a positive effect on the target.
2. `Type of Travel`: The type of business travel has a positive effect on the target.
3. `Customer Type`: Loyal customers have a positive effect on the target.
4. `Total Delay`: A small total delay has a positive effect on the target.
5. Overall, the higher the satisfaction value of each feature has a positive effect on the target, but there are some that actually have a negative effect such as `Gate Location` and `Ease of Online Booking`.

<br />

Conclusion & Recommendation
---------------------------
**Confusion Matrix :**

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Confusion%20Matrix.PNG)

General information about ticket:
- Average ticket price per person = 116 USD (not separated by `class`)
- Campaign/Incentive Expense = 6 USD (4.5%-5% of revenue, we take the highest range)

Summary:
- Retained customer = 1.339.452 (TN)
- Campaign Cost = 69.282 (TN) + 3.066 (FN) = 72.348 USD
- Potential Loss = 45.124 USD (FP)
- Potential Save = 50.964 USD (TP)
- Total possible income after campaign cost = 1.273.034 USD

<br />

**Conclusion :**

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Classification%20Report%20CatBoost.PNG)

Things that can be concluded based on the results of the classification report:
- Based on `Recall`, there are 98% of customers from predicted dissatisfied customers who actually need to be given incentive, and there are 94% of customers from predicted satisfied customers who actually do not need to be given incentive.
- Based on `Precision`, we can predict 96% of dissatisfied customers correctly, and predict 97% of satisfied customers correctly.
- Based on `Accuracy`, we can accurately predict 96% of customers who are satisfied and dissatisfied.
- Based on the `ROC AUC`, we can distinguish the two classes (satisfied and dissatisfied/neutral) almost perfectly.

<br />

**Recommendation :**

Actions that can be taken to help retain customers:
- Provide incentives for `inflight wifi service` features, such as:
  - Offers a more affordable wi-fi price on online booking and a normal price onsite to reduce extra costs.
  - Offers 15-30 minutes of free wi-fi for every Eco/Eco Plus class customers who fly above a certain limit (3000 miles / 6 hours) or experience delays above a certain limit (60 minutes)
- Based on EDA, it can be seen that the most `Type of Travel` is business travel, which is 69% of the total customers. Therefore, we can develop an existing B2B campaign to provide an upgrade from the Eco/Eco Plus class to the Business class at a special price, because Business class tends to have a higher level of satisfaction than other classes.
- Based on EDA, we can see that customer 'loyalty' greatly affects customer grouping in terms of age, class, and type of travel. Generally, bonus mileage can help customers to fly more often, so we can vary the incentives they can get from using their mileage, such as lounges at the airport or hotels at their destination.
- Review and develop the 'Online Boarding' system, especially in Eco/Eco Plus classes, because Business class has a higher level of satisfaction.
- Based on EDA, it can be seen that Business and Eco classes are evenly divided (48:44), but more dissatisfied customers are in Economy class (36% of 44%) than Business class (14% of 48%). From there, we can prioritize handling customer satisfaction in the business class.
- Provide incentives for `Age` features, such as:
  - Customers who are relatively younger (under 30) can be targeted on becoming loyal customers by providing more bonus mileage, given that this age group tends to travel more often.
  - Customers who are relatively older (above 60) can be targeted on upgrading to Business Class by providing a promo price as a Senior Discount Campaign.
- It can be seen that there are 3 services in the 'Airport experience' group that have a fairly high impact, so we can focus our resources on improving the quality of services at the airport.

<br />

Things that can be done to develop the project and model to be even more reliable:
- Try other ML algorithms and models, then do a better hyperparameter tuning.
- Conduct more detailed surveys on features that have a significant effect on customer satisfaction, such as `Online Boarding` and `Inflight wifi service` as to improve model performance.

<br />

**Limitation :**

Model paling baik digunakan dengan beberapa batas nilai feature, seperti:
- `Age`: 7-85 years old
- `Flight Distance`: 31-4.983 miles
- `Departure/Arrival Delay`: 38-1.592 minutes
