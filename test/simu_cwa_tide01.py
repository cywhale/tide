import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from scipy import signal
import time
import random
import os, glob
from dotenv import load_dotenv

load_dotenv()

# Global test counter
test_sid = 5
target = 'CWA'
target_token = os.getenv(f"{target}_TOKEN")
fetch_mode = 'reuse_all' #'exclude-station-list' #reuse_data_odb # 'reuse_stations_force-update-data' 
Number_of_Testing = 30
ext_start_date = '' #'2024-07-04T00:00:00'  # Specify an external start date if needed
metadata = None
test_dir = 'test/'

# Convert UNIX timestamp to UTC datetime
def unix_to_utc(unix_timestamp):
    return datetime.fromtimestamp(unix_timestamp, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')

# Convert GMT+8 to UTC
def gmt8_to_utc(gmt8_datetime, tz=timezone.utc):
    gmt8_dt = datetime.strptime(gmt8_datetime, "%Y-%m-%dT%H:%M:%S%z")
    utc_dt = gmt8_dt.astimezone(tz) #pytz.timezone('Asia/Tokyo')) # 'US/Pacific'))
    return utc_dt.strftime('%Y-%m-%dT%H:%M:%S')

# Fetch data from Tide API
def fetch_tide_data(lon, lat, start, end):
    base_url = "https://eco.odb.ntu.edu.tw/api/tide"
    params = {
        "lon0": lon,
        "lat0": lat,
        "start": start,
        "end": end
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200 or not response.json():
        return {}
    return response.json()

def get_testbench_stations():
    testbench_stations = set()
    metadata_files = glob.glob(f'{test_dir}testbench_*_metadata_{target.lower()}_*.json')
    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as meta_file:
            metadatax = json.load(meta_file)
            testbench_stations.update(metadatax["selected_stations"])
    print("Testbench has these stations: ", testbench_stations)        
    return testbench_stations

def get_cwa_stations(start_date):
    global Number_of_Testing, fetch_mode, test_sid, metadata, target
    stations_file_path = f"{test_dir}stations_cwa.json"
    datex = start_date[0:10]  # "%Y-%m-%d"

    with open(stations_file_path, 'r') as file:
        stations_data = json.load(file)

    stations = stations_data["cwaopendata"]["Resources"]["Resource"]["Data"]["SeaSurfaceObs"]["Location"]

    if 'reuse_all' in fetch_mode:
        selected_station_ids = get_testbench_stations()
        selected_stations = [station for station in stations if station["Station"]["StationID"] in selected_station_ids]
    elif 'stations' in fetch_mode or 'reuse_data' in fetch_mode:
        with open(f'{test_dir}testbench_{datex}_metadata_{target.lower()}_{test_sid}.json', 'r') as meta_file:
            metadata = json.load(meta_file)
        print("Get stations from metadata: ", meta_file)
        selected_station_ids = metadata["selected_stations"]
        selected_stations = [station for station in stations if station["Station"]["StationID"] in selected_station_ids]
    else:
        excluded_stations = get_testbench_stations() if 'exclude-station-list' in fetch_mode else set()
        selected_stations = []
        retry_limit_counter = 0
        retry_limit = 100
        while len(selected_stations) < Number_of_Testing and retry_limit_counter < retry_limit:
            station = random.choice(stations)
            station_id = station["Station"]["StationID"]
            if station_id not in excluded_stations and station not in selected_stations:
                selected_stations.append(station)
            retry_limit_counter += 1

    return selected_stations
    
def fetch_cwa_data(station_id, start_date_str):
    base_url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-B0075-002"
    params = {
        "Authorization": target_token,
        "StationID": station_id,
        "WeatherElement": "TideHeight,TideLevel",
        "sort": "DataTime",
        "timeFrom": start_date_str
    }
    response = requests.get(base_url, params=params)
    return response.json()

def extract_cwa_heights(cwa_data, lon, lat):
    records = cwa_data["Records"]["SeaSurfaceObs"]["Location"][0]["StationObsTimes"]["StationObsTime"]
    station_id = cwa_data["Records"]["SeaSurfaceObs"]["Location"][0]["Station"]["StationID"]
    heights = []
    for record in records:
        if 'TideHeight' not in record["WeatherElements"] or record["WeatherElements"]["TideHeight"] == 'None':
            continue
        dt_utc = gmt8_to_utc(record["DateTime"])
        height = float(record["WeatherElements"]["TideHeight"])
        heights.append({"station_id": station_id, "longitude": lon, "latitude": lat, "timestamp_utc": dt_utc, "height": height})
    if not heights:  # Check if heights list is empty
        return pd.DataFrame()  # Return empty DataFrame
    heights_df = pd.DataFrame(heights)
    heights_df["timestamp_utc"] = pd.to_datetime(heights_df["timestamp_utc"])
    return heights_df

def filter_and_compare(target_df, tide_df):
    min_time = max(target_df["timestamp_utc"].min(), tide_df["timestamp"].min())
    max_time = min(target_df["timestamp_utc"].max(), tide_df["timestamp"].max())
    
    target_filtered = target_df[(target_df["timestamp_utc"] >= min_time) & (target_df["timestamp_utc"] <= max_time)]
    tide_filtered = tide_df[(tide_df["timestamp"] >= min_time) & (tide_df["timestamp"] <= max_time)]
    
    odb_interpolated = tide_filtered.set_index('timestamp').reindex(target_filtered['timestamp_utc']).interpolate()
    
    comparison_df = pd.DataFrame({
        "timestamp_utc": target_filtered['timestamp_utc'],
        "height_target": target_filtered['height'],
        "height_tide": odb_interpolated['height'].values
    })
    comparison_df["height_diff"] = comparison_df["height_target"] - comparison_df["height_tide"]
    
    return comparison_df

def calculate_phase_difference(signal1, signal2):
    correlation = signal.correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2), mode='full')
    lags = signal.correlation_lags(len(signal1), len(signal2), mode='full')
    lag = lags[np.argmax(correlation)]
    return lag

def save_metadata_and_data(selected_stations, start_date, cwa_heights, tide_df, first_station):
    global test_sid, fetch_mode
    datex = start_date[0:10] #"%Y-%m-%d"
    modex = 'a'
    
    metadatax = {
        "target": "CWA",
        "selected_stations": [station["Station"]["StationID"] for station in selected_stations],
        "start_date": start_date,
        "end_date": (datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S') + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S'),
        "timezone": 'GMT+8',
        "target_data": f'testbench_{datex}_data_cwa_{test_sid}.csv',
        "odb_data": f'testbench_{datex}_data_odb_{test_sid}.csv'
    }

    if 'reuse' not in fetch_mode and 'stations' not in fetch_mode:
        with open(f'{test_dir}testbench_{datex}_metadata_cwa_{test_sid}.json', 'w') as meta_file:
            json.dump(metadatax, meta_file, indent=4)

    if 'force-update-data' in fetch_mode:
        if first_station:
            modex = 'w'
        cwa_heights.to_csv(f"{test_dir}{metadatax['target_data']}", mode=modex, header=first_station, index=False)
    elif 'reuse' not in fetch_mode and 'stations' not in fetch_mode:
        cwa_heights.to_csv(f"{test_dir}{metadatax['target_data']}", mode=modex, header=not os.path.exists(f"{test_dir}{metadatax['target_data']}"), index=False)
            
    if 'force-update-data' in fetch_mode:
        if first_station:
            modex = 'w'
        tide_df.to_csv(f"{test_dir}{metadatax['odb_data']}", mode=modex, header=first_station, index=False)
    elif 'reuse' not in fetch_mode and 'stations' not in fetch_mode:
        tide_df.to_csv(f"{test_dir}{metadatax['odb_data']}", mode=modex, header=not os.path.exists(f"{test_dir}{metadatax['odb_data']}"), index=False)

def extract_data_from_csv(selected_stations):
    # station_stats = []  
    cwa_heights = pd.DataFrame()
    tide_df = pd.DataFrame()      
    if len(selected_stations) > 0:
        data_files = glob.glob(f'{test_dir}testbench_*_data_{target.lower()}_*.csv')
        for data_file in data_files:
            new_data = pd.read_csv(data_file)
            #print("new data:", new_data)
            #print("all cwa_heights: ", cwa_heights)
            cwa_heights = pd.concat([cwa_heights, new_data]).drop_duplicates(subset=["station_id", "timestamp_utc"])

        odb_files = glob.glob(f'{test_dir}testbench_*_data_odb_*.csv')
        for odb_file in odb_files:
            new_odb = pd.read_csv(odb_file)
            tide_df = pd.concat([tide_df, new_odb]).drop_duplicates(subset=["longitude", "latitude", "timestamp"])

    return cwa_heights, tide_df

     
def plot_statistics(stats):
    station_ids = [stat["station_id"] for stat in stats]
    amp_diff_means = [stat["amp_diff_mean"] for stat in stats]
    amp_diff_stds = [stat["amp_diff_std"] for stat in stats]
    phase_diff_hours = [stat["phase_diff_hours"] for stat in stats]

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.errorbar(station_ids, amp_diff_means, yerr=amp_diff_stds, fmt='o', ecolor='r', capsize=5, label='Amplitude Difference')
    plt.xticks(rotation=90)
    plt.xlabel('Station ID')
    plt.ylabel('Amplitude Difference (m)')
    plt.title('Amplitude Differences Across Stations')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(station_ids, phase_diff_hours, 'o-', label='Phase Difference (hours)')
    plt.xticks(rotation=90)
    plt.xlabel('Station ID')
    plt.ylabel('Phase Difference (hours)')
    plt.title('Phase Differences Across Stations')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main_batch(selected_stations, start_date):
    global test_sid, metadata, fetch_mode, target
    station_stats = []
    first_station = True

    if ('reuse' in fetch_mode or 'stations' in fetch_mode) and 'reuse_all' not in fetch_mode and metadata is None:
        print("Error: check why metadata is None in fetch_mode: ", fetch_mode)
        return

    if 'reuse_all' in fetch_mode:
        all_cwa, all_tide = extract_data_from_csv(selected_stations)

    for station in selected_stations:
        station_id = station["Station"]["StationID"]
        station_name = station["Station"]["StationName"]
        lon = float(station["Station"]["StationLongitude"])
        lat = float(station["Station"]["StationLatitude"])
        print(f"Testing station: {station_id}, with coord: {lon}, {lat}")

        if 'reuse_all' in fetch_mode:
            cwa_heights = all_cwa[(all_cwa["station_id"] == station_id) & (all_cwa["longitude"] == lon) & (all_cwa["latitude"] == lat)]
            tide_df = all_tide[(all_tide["longitude"] == lon) & (all_tide["latitude"] == lat)]            
            if cwa_heights.empty or tide_df.empty:
                continue            
        else:    
            cwa_data = fetch_cwa_data(station_id, start_date)
            if 'reuse_data' in fetch_mode:
                print(f"Reuse {target} data from: ", metadata["target_data"])
                cwa_heights = pd.read_csv(f"{test_dir}{metadata['target_data']}")
                cwa_heights["timestamp_utc"] = pd.to_datetime(cwa_heights["timestamp_utc"])
                cwa_heights = cwa_heights[(cwa_heights["station_id"] == station_id)] # & (cwa_heights["longitude"] == lon) & (cwa_heights["latitude"] == lat)]
            else:
                cwa_data = fetch_cwa_data(station_id, start_date)
                if 'SeaSurfaceObs' not in cwa_data['Records'] or not cwa_data['Records']['SeaSurfaceObs']['Location'][0]['StationObsTimes']['StationObsTime']:
                    continue
                # print("Get CWA data: ", cwa_data)
                cwa_heights = extract_cwa_heights(cwa_data, lon, lat)
                if cwa_heights.empty:
                    continue

            if 'reuse' in fetch_mode and 'odb' in fetch_mode:
                print("Reuse ODB data from: ", metadata["odb_data"])
                tide_df = pd.read_csv(f"{test_dir}{metadata['odb_data']}")
                tide_df["timestamp"] = pd.to_datetime(tide_df["timestamp"])
                #tide_df = tide_df[(pd.to_numeric(tide_df["longitude"], errors='coerce').round(5) == float(lon)) & (pd.to_numeric(tide_df["latitude"], errors='coerce').round(5) == float(lat))]
                tide_df = tide_df[(tide_df["longitude"] == lon) & (tide_df["latitude"] == lat)]
                if tide_df.empty:
                    print("Warning: ODB reused data is empty for ", station_id, lon, lat)
                    continue           
            else:
                start_utc = cwa_heights["timestamp_utc"].min().strftime('%Y-%m-%dT%H:%M:%S')
                end_utc = (cwa_heights["timestamp_utc"].max() + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
                tide_data = fetch_tide_data(lon, lat, start_utc, end_utc)
                if not tide_data:
                    continue
        
                # Extract heights from Tide API
                tide_times = pd.to_datetime(tide_data["time"])
                tide_heights = np.array(tide_data["z"]) / 100  # Convert cm to m
                tide_df = pd.DataFrame({"timestamp": tide_times, "height": tide_heights})
                tide_df["longitude"] = lon
                tide_df["latitude"] = lat
        
        # print("cwa_heights: ", cwa_heights)    
        # print("Get tide_df: ", tide_df)    
        comparison_df = filter_and_compare(cwa_heights, tide_df)
        phase_diff = calculate_phase_difference(comparison_df["height_target"], comparison_df["height_tide"])
        time_lag = phase_diff / 60.0  # Convert from seconds to hours

        amp_diff_mean = comparison_df["height_diff"].mean()
        amp_diff_std = comparison_df["height_diff"].std()

        station_stats.append({
            "station_id": f"{station_id} {station_name}",
            "amp_diff_mean": amp_diff_mean,
            "amp_diff_std": amp_diff_std,
            "phase_diff_hours": time_lag
        })

        if 'reuse_all' not in fetch_mode:
            save_metadata_and_data(selected_stations, start_date, cwa_heights, tide_df, first_station)
            first_station = False  # Update flag after the first write
        
        if 'reuse' not in fetch_mode:
            # Pause to avoid hitting rate limit
            time.sleep(0.5)
    
    return station_stats

now = datetime.now(timezone.utc) - timedelta(1) + timedelta(hours=8)
start_date = datetime.strptime(ext_start_date, '%Y-%m-%dT%H:%M:%S') if ext_start_date != '' and ('reuse' in fetch_mode or 'stations' in fetch_mode) else now.strftime('%Y-%m-%dT00:00:00')
print("Testing start_date: ", start_date)
selected_stations = get_cwa_stations(start_date)
print("Testing start, select stations: ", [sta["Station"]["StationName"] for sta in selected_stations])
stats = main_batch(selected_stations, start_date)
plot_statistics(stats)
