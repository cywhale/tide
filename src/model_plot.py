import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def plot_model_mask(mask):
    # Create the lon and lat values (assuming you haven't already)
    lon_values = np.linspace(0, 360, 10800)
    lat_values = np.linspace(-90, 90, 5400)

    # Extracting coordinates where the mask is True
    lon_masked = lon_values[np.where(mask)[1]]
    lat_masked = lat_values[np.where(mask)[0]]

    # Create a figure with a more square aspect ratio
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the Basemap object with Robinson projection
    # Note that if the 'merc' (Mercator) projection being used for the entire world's extent,
    # when a Mercator projection is used for a global scale (from -90 to 90 degrees latitude),
    # it exaggerates areas at the poles, resulting in a stretched appearance.
    m = Basemap(projection='robin', lat_0=0, lon_0=180, ax=ax, resolution='c')

    # Convert masked lon, lat to map projection coordinates
    x_masked, y_masked = m(lon_masked, lat_masked)

    # Add coastlines
    m.drawcoastlines()
    # Add ticks for meridians and parallels
    parallels = range(-90, 90, 20)
    meridians = range(0, 360, 40)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.5, fontsize=10)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.5, fontsize=10)

    # Plot the masked points
    m.scatter(x_masked, y_masked, s=0.1, color='red', marker='.',
              label='Masked Points')  # s is the size, adjust if needed

    # Optional: legend to indicate the masked points
    ax.legend(loc='upper right')
    ax.set_title('Masked Bathymetry Data Points')
    plt.show()


# Plot tidal height
def plot_tide_level(tide, dtime, miny=None, maxy=None, intervaly=None):
    miny = min(tide.data) if miny is None else miny
    maxy = max(tide.data) if maxy is None else maxy
    intervaly = (maxy-miny)/8 if intervaly is None else intervaly
    miny = int(miny)
    maxy = int(maxy)
    intervaly = int(intervaly)

    plt.figure(figsize=(18, 4))
    plt.plot(dtime, tide.data, marker='o')
    plt.xlabel('Time')
    plt.ylabel('Tidal height')
    plt.yticks(list(range(miny, maxy+1, intervaly)),
               [str(i) for i in range(miny, maxy+1, intervaly)])
    plt.grid()
    plt.show()


def plot_current(x, y, u, v, mag, label_time):
    # Calculate magnitude of the current
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotting, note the "angles" and "scale" parameters
    plt.colorbar(ax.quiver(x, y, u, v, mag,
                           angles='xy', scale_units='xy',
                           scale=3, pivot='middle', width=0.003, cmap='jet'),
                 ax=ax, label='Current speed')

    ax.set_title('Tidal Currents at ' + str(label_time))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.show()


def plot_current_map(x, y, u, v, mag, label_time):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the Basemap object
    lon_min, lon_max, lat_min, lat_max = np.min(
        x), np.max(x), np.min(y), np.max(y)

    # Round the min/max to the nearest 5 for setting ticks
    lon_start = int(5 * (lon_min // 5))
    # +1 to ensure the upper bound is included
    lon_end = int(5 * (lon_max // 5 + 1))

    lat_start = int(5 * (lat_min // 5))
    # +1 to ensure the upper bound is included
    lat_end = int(5 * (lat_max // 5 + 1))

    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                llcrnrlon=lon_min, urcrnrlon=lon_max, lat_ts=20, ax=ax, resolution='i')

    # Convert x, y to map projection coordinates
    x_map, y_map = m(x, y)
    # Add coastlines
    m.drawcoastlines()

    # Add ticks for meridians and parallels
    parallels = range(lat_start, lat_end, 5)
    meridians = range(lon_start, lon_end, 5)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.5, fontsize=10)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.5, fontsize=10)

    # Plotting the tidal currents
    quiver = m.quiver(x_map, y_map, u, v, mag, angles='xy', scale_units='dots',
                      scale=0.1, pivot='middle', width=0.003, cmap='jet')
    plt.colorbar(quiver, ax=ax, label='Current speed')

    ax.set_title('Tidal Currents at ' + str(label_time))
    ax.set_xlabel('Longitude', labelpad=30)  # Adjust labelpad as needed
    ax.set_ylabel('Latitude', labelpad=30)   # Adjust labelpad as needed

    plt.show()
