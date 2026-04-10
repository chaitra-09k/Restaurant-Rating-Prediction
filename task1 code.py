import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("Dataset .csv")

data = data[['Average Cost for two', 'Votes', 'Price range',
             'Has Table booking', 'Has Online delivery',
             'Aggregate rating']]

data['Has Table booking'] = data['Has Table booking'].map({'Yes':1, 'No':0})
data['Has Online delivery'] = data['Has Online delivery'].map({'Yes':1, 'No':0})

data = data.dropna()

X = data.drop('Aggregate rating', axis=1)
y = data['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)

print("Random Forest Error:", error)

print("\nSample Predictions:")
for i in range(5):
    print("Predicted:", round(predictions[i],2), "| Actual:", y_test.iloc[i])

# ----------------- USER INPUT -----------------

full_data = pd.read_csv("Dataset .csv")


city_input = input("\nEnter city (or press Enter to skip): ")
votes_input = input("Enter minimum votes (or press Enter to skip): ")
delivery_input = input("Online delivery? (Yes/No or press Enter to skip): ")
booking_input = input("Table booking? (Yes/No or press Enter to skip): ")

filtered_data = full_data.copy()

# Apply filters one by one

# City filter
if city_input:
    filtered_data = filtered_data[
        filtered_data['City'].str.lower() == city_input.lower()
    ]

# Votes filter
if votes_input:
    filtered_data = filtered_data[
        filtered_data['Votes'] >= int(votes_input)
    ]

# Delivery filter
if delivery_input:
    filtered_data = filtered_data[
        filtered_data['Has Online delivery'].str.lower() == delivery_input.lower()
    ]

# Table booking filter
if booking_input:
    filtered_data = filtered_data[
        filtered_data['Has Table booking'].str.lower() == booking_input.lower()
    ]

# Check empty
if filtered_data.empty:
    print("\nNo matching restaurants found. Try different filters.")
else:
    # Select needed columns
    filtered_data = filtered_data[['Restaurant Name', 'City',
                                   'Average Cost for two', 'Votes', 'Price range',
                                   'Has Table booking', 'Has Online delivery']]

    # Convert Yes/No to numbers
    filtered_data['Has Table booking'] = filtered_data['Has Table booking'].map({'Yes':1, 'No':0})
    filtered_data['Has Online delivery'] = filtered_data['Has Online delivery'].map({'Yes':1, 'No':0})

    filtered_data = filtered_data.dropna()

    # Features
    X_new = filtered_data[['Average Cost for two', 'Votes', 'Price range',
                           'Has Table booking', 'Has Online delivery']]

    # Predict
    filtered_data['Predicted Rating'] = model.predict(X_new)

    # Sort
    filtered_data = filtered_data.sort_values(by='Predicted Rating', ascending=False)

    print("\nTop Restaurants:")
    print(filtered_data[['Restaurant Name', 'City', 'Predicted Rating']].head(10))