import json
import pickle
import numpy as np
import pandas as pd

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    global __data_columns, __model

    location = location.lower()  
    x = np.zeros(len(__data_columns))

    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in __data_columns:
        loc_index = __data_columns.index(location)  
        x[loc_index] = 1

    input_df = pd.DataFrame([x], columns=__data_columns)

    return round(__model.predict(input_df)[0], 2)  # returns estimated price in lakh rupees


def get_location_names():
    return __locations

def load_saved_artifacts():
    global __data_columns
    global __locations

    with open("server/artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open("server/artifacts/bengaluru_house_data.pickle", 'rb') as f:
        __model = pickle.load(f)



if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())

### test values    
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))
    print(get_estimated_price('Hebbal', 1000, 2, 2))

