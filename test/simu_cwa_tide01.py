import json
import requests
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from timezonefinder import TimezoneFinder
import pytz
import time
import random
import os, glob
from dotenv import load_dotenv

load_dotenv()

# Global test counter
test_sid = 12
target = 'NOAA' # 'CWA', 'NOAA'
target_token = os.getenv(f"{target}_TOKEN")
fetch_mode = 'reuse_all' #'reuse_all' #'exclude-station-list' #reuse_data_odb # 'reuse_stations_force-update-data' 
Number_of_Testing = 30
ext_start_date = '2024-07-08T00:00:00' #2024-07-07T00:00:00' #'2024-07-04T00:00:00'  # Specify an external start date if needed
metadata = None
test_dir = 'test/'

station_timezone = 'Asia/Taipei'
now = datetime.now(timezone.utc) - timedelta(1) + timedelta(hours=8)
local_start_date = datetime.strptime(ext_start_date, '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S') if ext_start_date != '' and ('reuse' in fetch_mode or 'stations' in fetch_mode) else now.strftime('%Y-%m-%dT00:00:00')

if target == 'NOAA':
    skip_stations = ['8770822', '8775132', '8770613', '8770520', '8770475', '8773259', '8723214', '8775241', '8773767'] # "Aransas, Aransas Pass" station has very weird waveforms that can hardly compared by cross correlation
    # 8723214: Virginia Key, Biscayne Bay has some empty tide datas, cannnot be compared by correlation
    # 8770822 (Texas Point, Sabine Pass), 8775132 (La Quinta Ch. North), 8770613 (Morgans Point), 8770520 (Rainbow Bridge), 8770475 (Port Arthur), 8773259: Port Lavaca, are bad in waveform cannot be considered as periodic, that cannot use to compare
else:
    skip_stations = ['']


# Find timezone for NOAA 'LST_LDT'
def find_timezone(lon, lat):
    tf = TimezoneFinder()
    tz_str = tf.timezone_at(lng=lon, lat=lat)
    return tz_str

# Adjust the start date based on the station's time zone
def get_local_start_date(local_timezone=None, station_lon=None, station_lat=None):
    if local_timezone is None or not local_timezone:
        local_timezone = find_timezone(station_lon, station_lat)
        if not local_timezone:
            raise ValueError("Timezone not found for given coordinates: ", station_lon, station_lat)
    local_tz = pytz.timezone(local_timezone)
    now = datetime.now(local_tz) - timedelta(1)
    start_date = now.strftime('%Y-%m-%dT00:00:00')
    return start_date

# Convert UNIX timestamp to UTC datetime
def unix_to_utc(unix_timestamp):
    return datetime.fromtimestamp(unix_timestamp, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')

def local_to_utc(local_datetime, local_timezone='Asia/Taipei', tz=timezone.utc, format="%Y-%m-%dT%H:%M:%S%z"):
    # Handle both naive and aware datetime strings
    try:
        local_tz = pytz.timezone(local_timezone)
        local_dt = local_tz.localize(datetime.strptime(local_datetime, format))  
    except ValueError:
        local_dt = datetime.strptime(local_datetime, format)  # Try to parse as aware datetime
    utc_dt = local_dt.astimezone(tz)
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

def get_testbench_stations(local_date=None):
    testbench_stations = set()
    if local_date is not None:
        metadata_files = glob.glob(f'{test_dir}testbench_{local_date}_metadata_{target.lower()}_*.json')
    else:    
        metadata_files = glob.glob(f'{test_dir}testbench_*_metadata_{target.lower()}_*.json')
    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as meta_file:
            metadatax = json.load(meta_file)
            testbench_stations.update(metadatax["selected_stations"])
    print("Testbench has these stations: ", testbench_stations)        
    return testbench_stations

def get_stations():
    global Number_of_Testing, fetch_mode, test_sid, metadata, target, local_start_date, skip_stations
    stations_file_path = f"{test_dir}stations_{target.lower()}.json"
    datex = local_start_date[0:10]  # "%Y-%m-%d"

    with open(stations_file_path, 'r') as file:
        stations_data = json.load(file)

    if target == "CWA":
        stations = stations_data["cwaopendata"]["Resources"]["Resource"]["Data"]["SeaSurfaceObs"]["Location"]
    else:
        stations = [station for station in stations_data["portsStationList"] if station["waterlevel"]] 

    if 'reuse_all' in fetch_mode:
        selected_station_ids = get_testbench_stations()
        if target == "CWA":
            selected_stations = [station for station in stations if station["Station"]["StationID"] in selected_station_ids and station["Station"]["StationID"] not in skip_stations]
        else:
            selected_stations = [station for station in stations if station["stationID"] in selected_station_ids and station["stationID"] not in skip_stations]   
        
    elif 'stations' in fetch_mode or 'reuse_data' in fetch_mode:
        with open(f'{test_dir}testbench_{datex}_metadata_{target.lower()}_{test_sid}.json', 'r') as meta_file:
            metadata = json.load(meta_file)
        print("Get stations from metadata: ", meta_file)
        selected_station_ids = metadata["selected_stations"]
        if target == "CWA":
            selected_stations = [station for station in stations if station["Station"]["StationID"] in selected_station_ids and station["Station"]["StationID"] not in skip_stations]
        else:    
            selected_stations = [station for station in stations if station["stationID"] in selected_station_ids and station["stationID"] not in skip_stations]

    else:
        excluded_stations = get_testbench_stations(datex) if 'exclude-station-list' in fetch_mode else (
                            get_testbench_stations() if 'exclude-all-list' in fetch_mode else set())
        selected_stations = []
        retry_limit_counter = 0
        retry_limit = 100
        while len(selected_stations) < Number_of_Testing and retry_limit_counter < retry_limit:
            station = random.choice(stations)
            if target == "CWA":
                station_id = station["Station"]["StationID"]
            else:
                station_id = station["stationID"]    
            if station_id not in excluded_stations and station_id not in skip_stations and station not in selected_stations:
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

def fetch_noaa_data(station_id, start_date_str):
    startx = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%S')   
    end_strx = (startx + timedelta(days=1)).strftime('%Y%m%d')
    start_strx = startx.strftime('%Y%m%d')
    print(f"Fetch NOAA {start_strx}-{end_strx} for station {station_id}")
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "product": "water_level",
        "begin_date": start_strx,
        "end_date": end_strx,
        "datum": "MSL",
        "station": station_id,
        "time_zone": "LST_LDT",
        "units": "metric",
        "format": "json",
        "application": "NOS.COOPS.TAC.COOPSMAP"
    }
    response = requests.get(base_url, params=params)
    return response.json()

def extract_noaa_heights(noaa_data, local_timezone):
    print("Local timezone when extract NOAA height: ", local_timezone)
    station_id = noaa_data["metadata"]["id"]
    lon = float(noaa_data["metadata"]["lon"])
    lat = float(noaa_data["metadata"]["lat"])
    heights = []
    for record in noaa_data["data"]:
        if record["v"] == '' or record["v"] is None:
            break  # Stop processing the current time series if a missing value is encountered
        dt_utc = local_to_utc(record["t"], local_timezone=local_timezone, format="%Y-%m-%d %H:%M")
        height = float(record["v"])
        heights.append({"station_id": station_id, "longitude": lon, "latitude": lat, "timestamp_utc": dt_utc, "height": height})
    heights_df = pd.DataFrame(heights)
    heights_df["timestamp_utc"] = pd.to_datetime(heights_df["timestamp_utc"], errors='coerce')
    return heights_df

def extract_cwa_heights(cwa_data, lon, lat):
    records = cwa_data["Records"]["SeaSurfaceObs"]["Location"][0]["StationObsTimes"]["StationObsTime"]
    station_id = cwa_data["Records"]["SeaSurfaceObs"]["Location"][0]["Station"]["StationID"]
    heights = []
    for record in records:
        if 'TideHeight' not in record["WeatherElements"] or record["WeatherElements"]["TideHeight"] == 'None':
            continue
        dt_utc = local_to_utc(record["DateTime"])
        height = float(record["WeatherElements"]["TideHeight"])
        heights.append({"station_id": station_id, "longitude": lon, "latitude": lat, "timestamp_utc": dt_utc, "height": height})
    if not heights:  # Check if heights list is empty
        return pd.DataFrame()  # Return empty DataFrame
    heights_df = pd.DataFrame(heights)
    heights_df["timestamp_utc"] = pd.to_datetime(heights_df["timestamp_utc"], errors='coerce')
    return heights_df

def filter_and_compare(target_df, tide_df, testing=False):
    min_time = max(target_df["timestamp_utc"].min(), tide_df["timestamp"].min())
    max_time = min(target_df["timestamp_utc"].max(), tide_df["timestamp"].max())
    
    target_filtered = target_df[(target_df["timestamp_utc"] >= min_time) & (target_df["timestamp_utc"] <= max_time)]
    tide_filtered = tide_df[(tide_df["timestamp"] >= min_time) & (tide_df["timestamp"] <= max_time)]
    if testing:
        print("target_df: ", target_df)
        print("tide_df: ", tide_df)
        print("min/max time:", min_time, max_time)
        print("target_df filtered: ", target_filtered)
    odb_interpolated = tide_filtered.set_index('timestamp').reindex(target_filtered['timestamp_utc']).interpolate()
    if testing:
        print("odb interpolated: ", odb_interpolated)
    
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

def save_metadata_and_data(selected_stations, start_date, target_heights, tide_df, first_station):
    global test_sid, fetch_mode, target, station_timezone
    datex = start_date[0:10]
    modex = 'a'
    if target == 'CWA':
        selected_station_ids = [station["Station"]["StationID"] for station in selected_stations]
    else:
        selected_station_ids = [station["stationID"] for station in selected_stations]
    
    metadatax = {
        "target": target,
        "selected_stations": selected_station_ids,
        "start_date": start_date,
        "end_date": (datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S') + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S'),
        "timezone": station_timezone,
        "target_data": f'testbench_{datex}_data_{target.lower()}_{test_sid}.csv',
        "odb_data": f'testbench_{datex}_data_odb_{test_sid}.csv'
    }

    if 'reuse' not in fetch_mode and 'stations' not in fetch_mode:
        with open(f'{test_dir}testbench_{datex}_metadata_{target.lower()}_{test_sid}.json', 'w') as meta_file:
            json.dump(metadatax, meta_file, indent=4)

    if 'force-update-data' in fetch_mode:
        if first_station:
            modex = 'w'
        target_heights.to_csv(f"{test_dir}{metadatax['target_data']}", mode=modex, header=first_station, index=False)
    elif 'reuse' not in fetch_mode and 'stations' not in fetch_mode:
        target_heights.to_csv(f"{test_dir}{metadatax['target_data']}", mode='a', header=not os.path.exists(f"{test_dir}{metadatax['target_data']}"), index=False)
           

def extract_data_from_csv(selected_stations):
    target_heights = pd.DataFrame()
    tide_df = pd.DataFrame()      
    if len(selected_stations) > 0:
        data_files = glob.glob(f'{test_dir}testbench_*_data_{target.lower()}_*.csv')
        for data_file in data_files:
            testbench_sid = data_file.split('_')[-1].split('.')[0]  # Extract the testbench_sid from filename
            new_data = pd.read_csv(data_file, dtype={'station_id': 'string'})
            new_data['testbench_sid'] = testbench_sid
            target_heights = pd.concat([target_heights, new_data]).drop_duplicates(subset=["station_id", "timestamp_utc"])

        odb_files = glob.glob(f'{test_dir}testbench_*_data_odb_*.csv')
        for odb_file in odb_files:
            testbench_sid = odb_file.split('_')[-1].split('.')[0]  # Extract the testbench_sid from filename
            new_odb = pd.read_csv(odb_file)
            new_odb['testbench_sid'] = testbench_sid
            tide_df = pd.concat([tide_df, new_odb]).drop_duplicates(subset=["longitude", "latitude", "timestamp"])

    return target_heights, tide_df
    
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

def main_batch(selected_stations):
    global test_sid, metadata, fetch_mode, target, local_start_date, station_timezone
    station_stats = []
    first_station = True

    if ('reuse' in fetch_mode or 'stations' in fetch_mode) and 'reuse_all' not in fetch_mode and metadata is None:
        print("Error: check why metadata is None in fetch_mode: ", fetch_mode)
        return

    if 'reuse_all' in fetch_mode:
        all_target, all_tide = extract_data_from_csv(selected_stations)

    for station in selected_stations:
        if target == 'CWA':
            station_id = station["Station"]["StationID"]
            # station_name = station["Station"]["StationName"]
            lon = float(station["Station"]["StationLongitude"])
            lat = float(station["Station"]["StationLatitude"])
        else:
            station_id = station["stationID"]
            # station_name = station["label"]
            lon = float(station["lng"])
            lat = float(station["lat"])
            station_timezone = find_timezone(lon, lat)
            local_start_date = get_local_start_date(local_timezone=station_timezone)    
        print(f"Testing {target} stations: {station_id}, with coord: {lon}, {lat}, for start_date: {local_start_date} at timezone: {station_timezone}")

        if 'reuse_all' in fetch_mode:
            target_heights = all_target[(all_target["station_id"] == station_id) & (all_target["longitude"] == lon) & (all_target["latitude"] == lat)]
            tide_df = all_tide[(all_tide["longitude"] == lon) & (all_tide["latitude"] == lat)]            
            for sid in target_heights['testbench_sid'].unique():
                target_subset = target_heights[target_heights['testbench_sid'] == sid]
                tide_subset = tide_df[tide_df['testbench_sid'] == sid]
                if target_subset.empty or tide_subset.empty:
                    continue
                comparison_df = filter_and_compare(target_subset, tide_subset, testing=(station_id == '8638901'))
                if comparison_df.empty:
                    continue
                phase_diff = calculate_phase_difference(comparison_df["height_target"], comparison_df["height_tide"])
                time_lag = phase_diff / 60.0  # Convert from seconds to hours

                amp_diff_mean = comparison_df["height_diff"].mean()
                amp_diff_std = comparison_df["height_diff"].std()

                station_stats.append({
                    "station_id": f"{station_id}_{sid}",
                    "amp_diff_mean": amp_diff_mean,
                    "amp_diff_std": amp_diff_std,
                    "phase_diff_hours": time_lag
                })
        else:    
            if 'reuse_data' in fetch_mode:
                print(f"Reuse {target} data from: ", metadata["target_data"])
                target_heights = pd.read_csv(f"{test_dir}{metadata['target_data']}", dtype={'station_id': 'string'})
                target_heights["timestamp_utc"] = pd.to_datetime(target_heights["timestamp_utc"], errors='coerce')
                target_heights = target_heights[(target_heights["station_id"] == station_id)]
                if target_heights.empty:
                        continue 
                
            else:
                if target == 'CWA':
                    target_data = fetch_cwa_data(station_id, local_start_date)
                    if 'SeaSurfaceObs' not in target_data['Records'] or not target_data['Records']['SeaSurfaceObs']['Location'][0]['StationObsTimes']['StationObsTime']:
                        continue
                    target_heights = extract_cwa_heights(target_data, lon, lat)
                    if target_heights.empty:
                        continue
                else:
                    target_data = fetch_noaa_data(station_id, local_start_date)
                    if not target_data or 'data' not in target_data or not target_data['data']:
                        continue
                    target_heights = extract_noaa_heights(target_data, station_timezone)
                    if target_heights.empty:
                        continue                    

            if 'reuse' in fetch_mode and 'odb' in fetch_mode:
                print("Reuse ODB data from: ", metadata["odb_data"])
                tide_df = pd.read_csv(f"{test_dir}{metadata['odb_data']}")
                tide_df["timestamp"] = pd.to_datetime(tide_df["timestamp"], errors='coerce')
                tide_df = tide_df[(tide_df["longitude"] == lon) & (tide_df["latitude"] == lat)]
                if tide_df.empty:
                    print("Warning: ODB reused data is empty for ", station_id, lon, lat)
                    continue           
            else:
                start_utc = target_heights["timestamp_utc"].min().strftime('%Y-%m-%dT%H:%M:%S')
                end_utc = (target_heights["timestamp_utc"].max() + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
                tide_data = fetch_tide_data(lon, lat, start_utc, end_utc)
                if not tide_data:
                    continue
        
                tide_times = pd.to_datetime(tide_data["time"], errors='coerce')
                tide_heights = np.array(tide_data["z"]) / 100  # Convert cm to m
                tide_df = pd.DataFrame({"timestamp": tide_times, "height": tide_heights})
                tide_df["longitude"] = lon
                tide_df["latitude"] = lat
        
            # if 'reuse_all' not in fetch_mode:
            if station_id == '9454240':
                print("Debug target_height: ", target_heights)
                target_heights.to_csv(f"{test_dir}test_data_{station_id}.csv", mode='w', header=True, index=False)
                print("Debug tide_df: ", tide_df)           
                tide_df.to_csv(f"{test_dir}test_data_odb_for_{station_id}.csv", mode='w', header=True, index=False)
            comparison_df = filter_and_compare(target_heights, tide_df)
            phase_diff = calculate_phase_difference(comparison_df["height_target"], comparison_df["height_tide"])
            time_lag = phase_diff / 60.0  # Convert from seconds to hours

            amp_diff_mean = comparison_df["height_diff"].mean()
            amp_diff_std = comparison_df["height_diff"].std()

            station_stats.append({
                "station_id": f"{station_id}",
                "amp_diff_mean": amp_diff_mean,
                "amp_diff_std": amp_diff_std,
                "phase_diff_hours": time_lag
            })

            save_metadata_and_data(selected_stations, local_start_date, target_heights, tide_df, first_station)
            first_station = False  # Update flag after the first write
        
        if 'reuse' not in fetch_mode:
            # Pause to avoid hitting rate limit
            time.sleep(0.5)
    
    return station_stats

selected_stations = get_stations()
if target == 'CWA':
    print("Testing start, select stations: ", [sta["Station"]["StationName"] for sta in selected_stations])
else:
    print("Testing start, select stations: ", [sta["label"] for sta in selected_stations])
     
stats = main_batch(selected_stations)
plot_statistics(stats)
