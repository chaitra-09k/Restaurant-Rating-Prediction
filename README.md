#  Restaurant Rating Prediction

## Objective
Build a machine learning model to predict the aggregate rating of a restaurant based on various features.

## ⚙️ Features Used
- Average Cost for two
- Votes
- Price Range
- Has Table Booking
- Has Online Delivery

##  Model Used
- Random Forest Regressor

##  Approach
- Cleaned dataset and handled missing values
- Converted categorical values into numerical format
- Split data into training and testing sets
- Trained the model using Random Forest
- Evaluated using Mean Absolute Error (MAE)

##  Results
- Achieved MAE ≈ 0.23
- Model predictions are close to actual ratings

##  Sample Output
Predicted: 4.48 | Actual: 4.3  
Predicted: 3.37 | Actual: 3.4  

##  Conclusion
The model successfully predicts restaurant ratings using key features like cost, votes, and services. Feature selection plays an important role in improving accuracy.

##  Tech Stack
- Python
- Pandas
- Scikit-learn
