import csv,\
        matplotlib.pyplot as plt, \
        numpy as np, \
        time
from datetime import datetime as dt


def extract_trips(csv_path, trip_list):
    count = 0
    # Extract trips from csv, put them in an OrderedDict with proper field names
    with open(csv_path, newline='') as tripdata_csv:
        reader = csv.DictReader(tripdata_csv, delimiter=',', quotechar='|')
        for row in reader:
            count += 1
            #print(row)
            if not row['start_station'] == '3000' and not row['end_station'] == '3000':
                trip_list.append(row)
            if count % 10000 == 0:
                print(f'Seen {count!s} rows')
    #print(trip_list)


def extract_stations(csv_path, station_list):
    count = 0
    # Extract trips from csv, put them in an OrderedDict with proper field names
    with open(csv_path, newline='') as tripdata_csv:
        reader = csv.DictReader(tripdata_csv, delimiter=',', quotechar='|')
        for row in reader:
            count += 1
            #print(row)
            station_list.append(row)
            if count % 10000 == 0:
                print(f'Seen {count!s} rows')


def plot_trips(trip_list):
    start_lats = [float(trip['start_lat']) for trip in trip_list]
    start_lons = [float(trip['start_lon']) for trip in trip_list]
    end_lats = [float(trip['end_lat']) for trip in trip_list]
    end_lons = [float(trip['end_lon']) for trip in trip_list]

    plt.scatter(start_lats, start_lons, c='g', marker='>')
    plt.scatter(end_lats, end_lons, c='r', marker='8')

    start_latlons = [(start_lats[i], start_lons[i]) for i in range(len(trip_list))]
    end_latlons = [(end_lats[i], end_lons[i]) for i in range(len(trip_list))]

    thetas = [2*np.pi*i / len(trip_list) for i in range(len(trip_list))]
    r = 5
    offsets = [(r * np.cos(theta), r * np.sin(theta)) for theta in thetas]
    horizontal_alignments = ['left' if theta <= np.pi / 2 or theta > np.pi * 3 / 2 else 'right' for theta in thetas]
    vertical_alignments = ['bottom' if theta <= np.pi else 'top' for theta in thetas]

    for i, coord in enumerate(start_latlons + end_latlons):
        trip_idx = i % len(trip_list)

        plt.annotate(s=trip_idx,
                     xy=coord,
                     xytext=offsets[trip_idx],
                     textcoords='offset points',
                     horizontalalignment=horizontal_alignments[trip_idx],
                     verticalalignment=vertical_alignments[trip_idx]
                     )

    plt.show()


def count_trips_per_station(trips, outbound_station_counter, inbound_station_counter,
                            start_time_window=None, end_time_window=None, weekdays_only=False):
    timer_begin = time.time()
    has_window = start_time_window and end_time_window
    for trip in trips:
        trip_start = dt.strptime(trip['start_time'], '%m/%d/%Y %H:%M')
        trip_start_time = trip_start.time()
        is_weekday = trip_start.isoweekday() < 6
        if (not has_window or is_time_between(start_time_window, end_time_window, trip_start_time))\
                and (not weekdays_only or is_weekday):
            # If no window provided, always count the trip
            # If window provided, only count the trip if the start_time is in the window
            outbound_station_counter.update({trip['start_station'] : 1})
            inbound_station_counter.update({trip['end_station'] : 1})
    print('CTPS_Counter elapsed time:    ', str(time.time() - timer_begin))


def count_trips_per_station_defaultdict(trips, outbound_station_counter, inbound_station_counter,
                            start_time_window=None, end_time_window=None, weekdays_only=False):
    timer_begin = time.time()
    has_window = start_time_window and end_time_window
    for trip in trips:
        trip_start = dt.strptime(trip['start_time'], '%m/%d/%Y %H:%M')
        trip_start_time = trip_start.time()
        is_weekday = trip_start.isoweekday() < 6
        if (not has_window or is_time_between(start_time_window, end_time_window, trip_start_time))\
                and (not weekdays_only or is_weekday):
            # If no window provided, always count the trip
            # If window provided, only count the trip if the start_time is in the window
            outbound_station_counter[trip['start_station']] += 1
            inbound_station_counter[trip['end_station']] += 1
            sorted(outbound_station_counter.items())
            sorted(inbound_station_counter.items())
    print('CTPS_DefaultDict elapsed time: ', str(time.time() - timer_begin))


# Adapted from post by Joe Halloway on Stack Overflow: https://stackoverflow.com/a/10048290
def is_time_between(begin_time, end_time, check_time):
    if begin_time < end_time:
        return begin_time <= check_time <= end_time
    else:  # crosses midnight
        return check_time >= begin_time or check_time <= end_time


def plot_station_counts(station_counts):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter([station['morning_out'] for station in station_counts],
                [station['morning_in'] for station in station_counts])
    plt.xlabel('Morning Outbound Trips')
    plt.ylabel('Morning Inbound Trips')
    plt.axis('equal')
    plt.axis([0, 3000, 0, 3500])
    # plt.annotate()

    plt.subplot(122)
    plt.scatter([station['evening_out'] for station in station_counts],
                [station['evening_in'] for station in station_counts])
    plt.xlabel('Evening Outbound Trips')
    plt.ylabel('Evening Inbound Trips')
    plt.axis('equal')
    plt.axis([0, 3000, 0, 3500])

    # plt.axis([0, 3000, 0, 3000])
    plt.show()