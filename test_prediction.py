from src.predict import (
    predict_delay,
    get_prediction_label
)


sample_shipment = {

    "Asset_ID": "Truck_1",

    "Latitude": 20.5,

    "Longitude": 72.9,

    "Inventory_Level": 500,

    "Temperature": 30,

    "Humidity": 70,

    "Waiting_Time": 40,

    "User_Transaction_Amount": 5000,

    "User_Purchase_Frequency": 5,

    "Year": 2024,

    "Month": 6,

    "Day": 15,

    "Hour": 14,

    "Day_of_Week": 5,

    "Is_Weekend": 1,

    "Temperature_Risk": "High",

    "Humidity_Risk": "High",

    "Waiting_Risk": "Medium",

    "Inventory_Risk": "High"
}


result = predict_delay(
    sample_shipment
)


print("\nPrediction Result")
print("=================")

print(
    "Status:",
    get_prediction_label(
        result["prediction"]
    )
)

print(
    "Delay Probability:",
    f"{result['probability'] * 100:.2f}%"
)