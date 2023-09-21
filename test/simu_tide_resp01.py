import numpy as np
import polars as pl
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

def tide_to_output(tide, lon, lat, variables, format="list"):
    # Check if tide is all NaN values
    #if np.all(np.isnan(tide[var]) for var in variables): #only np.arrary work
    if all(np.all(np.isnan(np.array(tide[var])) if isinstance(tide[var], list) else np.isnan(tide[var])) for var in variables):   
        # Return an empty JSON response
        return {}
    
    # Generate longitude and latitude grids
    longitude, latitude = np.meshgrid(lon, lat)

    # Flatten the longitude and latitude grids
    longitude_flat = longitude.ravel()
    latitude_flat = latitude.ravel()

    # Create dictionary to hold results
    out_dict = {
        'longitude': longitude_flat,
        'latitude': latitude_flat,
    }

    # Flatten and add to dictionary the variables that are present
    for var in variables:
        out_dict[var] = tide[var]

    if format == 'list':
        return out_dict

    # Convert the dictionary to a Polars DataFrame (or other desired format)
    df = pl.DataFrame(out_dict)
    
    # Assuming you are using Polars, you can replace the above line with your actual code to convert to DataFrame

    # Return the data in the desired format
    return df  # Modify this part to return the data in the desired format

# Example usage:
#tide = {'z': np.array([np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])}  # Example input with all NaN values
tide = {'z': [np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}  # Example input with all NaN values, not np.array
lon = [1, 2, 3]
lat = [4, 5, 6]
variables = ['z']
response = tide_to_output(tide, lon, lat, variables, 'list')
print(response)
df = tide_to_output(tide, lon, lat, variables, 'df')
print(df)