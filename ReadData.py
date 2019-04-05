import os
import copy
import csv
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import OrderedDict, defaultdict
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from math import sqrt
from scipy.spatial.distance import cdist

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def extract_trips_dir(csv_dir, trip_list, valid_station_ids):
    csv_paths = []
    trip_list = []

    print(f'Extracting files from ', csv_dir)

    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            csv_paths.append(os.path.join(csv_dir, file))

    for csv_path in csv_paths:
        trip_list.extend(extract_trips(csv_path, [], valid_station_ids))

    print(f'Finished extracting files from ', csv_dir)
    return trip_list


def extract_trips(csv_path, trip_list, exclude_station_ids=None):
    if exclude_station_ids is None:
        exclude_station_ids = []
    count = 0
    # Extract trips from csv, put them in an OrderedDict with proper field names
    with open(csv_path, newline='') as tripdata_csv:
        print('Opening\t', csv_path)
        reader = csv.DictReader(tripdata_csv, delimiter=',', quotechar='"')
        for row in reader:
            count += 1

            start_station_id = row['start_station']
            end_station_id = row['end_station']
            # Only count rows for which there is a valid station
            if (start_station_id not in exclude_station_ids) and (end_station_id not in exclude_station_ids):
                trip_list.append(row)

    print('Closing\t', csv_path)
    print(f'Saw {count!s} rows')
    print(f'Extracted {len(trip_list)!s} trips')
    print('Ignored the following stations:')
    print(exclude_station_ids)
    return trip_list


def extract_stations_from_file(csv_path, station_list=None, exclude_station_ids=None):
    if station_list is None:
        station_list = []
    if exclude_station_ids is None:
        exclude_station_ids = []

    # Extract trips from csv, put them in an OrderedDict with proper field names
    with open(csv_path, newline='') as tripdata_csv:
        reader = csv.DictReader(tripdata_csv, delimiter=',', quotechar='"')
        for row in reader:
            if row['Station ID'] not in exclude_station_ids:
                station_list.append(row)
    return station_list




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
                            start_time_window=None, end_time_window=None, ok_days=None):
    timer_begin = time.time()
    has_window = start_time_window and end_time_window

    if ok_days is None:
        ok_days = [1, 2, 3, 4, 5, 6, 7]

    for trip in trips:
        trip_start = dt.datetime.strptime(trip['start_time'], DATETIME_FORMAT)
        trip_start_time = trip_start.time()

        is_in_time_window = not has_window or is_time_between(start_time_window, end_time_window, trip_start_time)
        is_in_days_window = trip_start.isoweekday() in ok_days

        if is_in_time_window and is_in_days_window:
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
        trip_start = dt.datetime.strptime(trip['start_time'], DATETIME_FORMAT)
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


def count_bucketed_trips(trips, resolution=60, ok_days=None, station_id=None):
    if ok_days is None:
        ok_days = [1, 2, 3, 4, 5, 6, 7]

    zero_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    num_buckets = int(1440 / resolution)
    if 1440 % resolution != 0:
        num_buckets += 1
    bucket_times = [zero_time + dt.timedelta(minutes=resolution * x) for x in range(num_buckets)]
    bucket_texts = [t.strftime('%H:%M') for t in bucket_times]
    start_hourly_counts = {k: 0 for k in bucket_texts}  #  defaultdict(int, bucket_texts)
    end_hourly_counts = {k: 0 for k in bucket_texts} #  defaultdict(int)
    total_hourly_counts = {k: 0 for k in bucket_texts}
    # bucket_size = 1440 / float(num_buckets)

    for trip in trips:
            trip_start = dt.datetime.strptime(trip['start_time'], DATETIME_FORMAT)
            is_in_days_window = trip_start.isoweekday() in ok_days

            if is_in_days_window:
                start_minute = minute_of_day(trip_start)
                bucket_number = int(start_minute / resolution)
                start_hourly_counts[bucket_texts[bucket_number]] += 1
                total_hourly_counts[bucket_texts[bucket_number]] += 1

                trip_end = dt.datetime.strptime(trip['end_time'], DATETIME_FORMAT)
                end_minute = minute_of_day(trip_end)
                bucket_number = int(end_minute / resolution)
                end_hourly_counts[bucket_texts[bucket_number]] += 1
                total_hourly_counts[bucket_texts[bucket_number]] += 1

    start_sum = sum(start_hourly_counts.values())
    end_sum = sum(end_hourly_counts.values())
    total_sum = sum(total_hourly_counts.values())

    for bucket in start_hourly_counts:
        count = start_hourly_counts[bucket]
        start_hourly_counts[bucket] = (count, count / start_sum)
    for bucket in end_hourly_counts:
        count = end_hourly_counts[bucket]
        end_hourly_counts[bucket] = (count, count / end_sum)
    for bucket in total_hourly_counts:
        count = total_hourly_counts[bucket]
        total_hourly_counts[bucket] = (count, count / total_sum)

    # Sort the dicts
    start_hourly_counts = OrderedDict(sorted(start_hourly_counts.items()))
    end_hourly_counts = OrderedDict(sorted(end_hourly_counts.items()))
    total_hourly_counts = OrderedDict(sorted(total_hourly_counts.items()))

    return start_hourly_counts, end_hourly_counts, total_hourly_counts


# Returns dict of {station_id: (start_hourly_counts, end_hourly_counts, total_hourly_counts)}
def count_bucketed_trips_by_station(trips, stations, resolution=60, ok_days=None):
    if ok_days is None:
        ok_days = [1, 2, 3, 4, 5, 6, 7]


    zero_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    num_buckets = int(1440 / resolution)

    if 1440 % resolution != 0:
        num_buckets += 1
    bucket_times = [zero_time + dt.timedelta(minutes=resolution * x) for x in range(num_buckets)]
    bucket_texts = [t.strftime('%H:%M') for t in bucket_times]

    start_hourly_counts = {k: 0 for k in bucket_texts}  #  defaultdict(int, bucket_texts)
    end_hourly_counts = {k: 0 for k in bucket_texts} #  defaultdict(int)
    total_hourly_counts = {k: 0 for k in bucket_texts}

    zero_hourly_counts = {k: 0 for k in bucket_texts}
    default_gen = lambda: [zero_hourly_counts] * 3
    # bucket_size = 1440 / float(num_buckets)

    # Making this a defaultdict ensures that we can still count even if we get an unexpected station ID
    counts_by_station = defaultdict(default_gen,
                                    {station['Station ID']: [copy.deepcopy(start_hourly_counts),
                                                             copy.deepcopy(end_hourly_counts),
                                                             copy.deepcopy(total_hourly_counts)]
                                     for station in stations})

    # Poor man's enum
    START = 0
    END = 1
    TOTAL = 2

    for trip in trips:
        trip_start = dt.datetime.strptime(trip['start_time'], DATETIME_FORMAT)
        is_in_days_window = trip_start.isoweekday() in ok_days

        if is_in_days_window:
            # Start of trip
            start_minute = minute_of_day(trip_start)
            bucket_number = int(start_minute / resolution)
            stn_id = trip['start_station']
            bkt_id = bucket_texts[bucket_number]

            counts_by_station[stn_id][START][bkt_id] += 1
            counts_by_station[stn_id][TOTAL][bkt_id] += 1

            # End of trip
            trip_end = dt.datetime.strptime(trip['end_time'], DATETIME_FORMAT)
            end_minute = minute_of_day(trip_end)
            bucket_number = int(end_minute / resolution)
            stn_id = trip['end_station']
            bkt_id = bucket_texts[bucket_number]

            counts_by_station[stn_id][END][bkt_id] += 1
            counts_by_station[stn_id][TOTAL][bkt_id] += 1

    complete_counts_by_station = defaultdict(default_gen)

    for stn_id, counts_list in counts_by_station.items():
        starts = counts_list[START]
        ends = counts_list[END]
        totals = counts_list[TOTAL]
        start_sum = sum(starts.values())
        end_sum = sum(ends.values())
        total_sum = sum(totals.values())
        # print(stn_id, starts)

        start_tuples = OrderedDict()
        end_tuples = OrderedDict()
        total_tuples = OrderedDict()

        for bucket in starts:
            absolute = starts[bucket]
            # print('abs', str(absolute), 'sum', str(start_sum))
            relative = absolute / start_sum if start_sum > 0 else 0
            start_tuples[bucket] = (absolute, relative)
        for bucket in ends:
            absolute = ends[bucket]
            relative = absolute / end_sum if end_sum > 0 else 0
            end_tuples[bucket] = (absolute, relative)
        for bucket in totals:
            absolute = totals[bucket]
            relative = absolute / total_sum if total_sum > 0 else 0
            total_tuples[bucket] = (absolute, relative)

        complete_counts_by_station[stn_id] = [start_tuples, end_tuples, total_tuples]
        # Sort the dicts
        starts = OrderedDict(sorted(start_hourly_counts.items()))
        ends = OrderedDict(sorted(end_hourly_counts.items()))
        totals = OrderedDict(sorted(total_hourly_counts.items()))

    return complete_counts_by_station


# Returns an int representing the the minute of the day this timestamp occurs in
def minute_of_day(timestamp):
    # print(timestamp)
    # print(timestamp.time())
    # print(timestamp.time().hour)
    return timestamp.time().hour * 60 + timestamp.time().minute


def plot_bucketed_count(bucketed_count, use_relative_count=False, group_n=4):
    xs = [x for x in range(len(bucketed_count))]
    # ws = [0.3] * len(bucketed_count)
    # print(f'DD of {len(bucketed_count)!s} buckets')
    # print(bucketed_count)
    # print(sorted(bucketed_count.items()))

    if use_relative_count:
        heights = [relative for _, (absolute, relative) in bucketed_count.items()]
    else:
        # print(bucketed_count.items())
        heights = [absolute for _, (absolute, relative) in bucketed_count.items()]

    # print(f'List of {len(heights)!s} heights')
    if group_n > 0:
        color_cycle = cycle(['#2D728F'] * group_n + ['#3B8EA5'] * group_n)  # Slightly different shades of blue
        colors = [next(color_cycle) for _ in range(len(bucketed_count))]
    else:
        colors = ['#2D728F'] * len(bucketed_count)

    plt.bar(xs, heights, width=0.8, color=colors, align='edge')
    plt.xticks(xs, bucketed_count.keys(), rotation=45)  # Uses bucket keys as tick labels

    # Only show every group_nth label
    for idx, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if group_n < 1 or idx % group_n != 0:
            label.set_visible(False)


def mse(A, B):
    return np.mean((B - A) ** 2)


def assemble_station_data(stn_ids, counts_by_stations, use_start_plus_end=True):
    station_data = []

    for stn_id in stn_ids:
        this_stn_data = []
        if use_start_plus_end:
            for count in counts_by_stations[stn_id][:2]:
                this_stn_data += [relative for absolute, relative in count.values()]
        else:  # Use  total instead of start_plus_end
            count = counts_by_stations[stn_id][2]
            this_stn_data = np.array([relative for absolute, relative in count.values()])

        station_data.append(this_stn_data)
    return station_data


def pca_dim_reduction(data, n_components=2):
    pca = PCA(n_components=n_components)
    scaled_data = StandardScaler().fit_transform(data)
    projected = pca.fit_transform(scaled_data)
    plt.scatter(projected[:, 0], projected[:, 1],
                alpha=0.5,
                cmap=plt.cm.get_cmap('tab10', 10))

    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()


def cluster_dbscan(data):
    # Based on https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html by scikit-learn developers
    # print('len of data =', len(data))
    scaled_data = StandardScaler().fit_transform(data)
    n_samples = scaled_data.shape[0]
    # print('len of scaled_data =', len(scaled_data))

    ### Don't need to do this on scaled data. Only do it on PCA projected data, see below
    # k means clustering w/ elbow plot to determine K
    # K = range(1, n_samples - 1, 10)
    # distortions = []
    # for k in K:
    #     kmeanModel = KMeans(n_clusters=k).fit(scaled_data)
    #     kmeanModel.fit(scaled_data)
    #
    #     distortions.append(sum(np.min(cdist(scaled_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / n_samples)
    #
    # for k, d in (zip(K, distortions)):
    #     print(k, d)
    # # Plot the elbow
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k for scaled_data')
    # plt.show()


    ###
    pca = PCA(n_components=2)
    projected_data = pca.fit_transform(scaled_data)

    ###
    # k means clustering w/ elbow plot to determine k for projected_data
    K = range(1, 15)
    distortions = []
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(projected_data)
        kmeanModel.fit(projected_data)

        distortions.append(sum(np.min(cdist(projected_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / n_samples)


    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k for projected_data')
    plt.show()
    # Visual inspection shows best elbow at k=4, eps = 1.07

    test_eps = distortions
    target_k = 12
    target_eps = test_eps[target_k - 1]

    plt.figure(figsize=(10, 10))
    n_rows = int(sqrt(len(test_eps))) + 1
    n_cols = int(sqrt(len(test_eps))) + 1
    n_subplots = n_rows * n_cols

    ###
    # Use this for loop to display DBSCAN with a range of eps values
    # target_labels = []
    # for i, eps in enumerate(test_eps):
    #     plt.subplot(n_rows, n_cols, i + 1)
    #     clustered = DBSCAN(eps=eps, min_samples=4).fit(projected_data)
    #     labels = clustered.labels_
    #     if i + 1 == target_k:
    #         target_labels = labels
    #     # Number of clusters in labels, ignoring noise if present.
    #     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    #     n_outliers = sum([(1 if label == -1 else 0) for label in labels])
    #     # Black removed and is used for noise instead.
    #     unique_labels = set(labels)
    #     unique_colors = [plt.cm.get_cmap('viridis')(each) for each in np.linspace(0, 1, n_clusters)]
    #     unique_colors.append((0, 0, 0, 1)) # Black is the last color, for label -1
    #
    #     colors = [unique_colors[label] for label in labels]
    #     # print(f'Found {len(labels)!s} labels')
    #
    #     plt.scatter(projected_data[:, 0], projected_data[:, 1],
    #                 c=colors, alpha=0.5)
    #
    #     plt.xlabel('component 1')
    #     plt.ylabel('component 2')
    #     plt.title(f'eps={round(eps, 2)!s}, clu={n_clusters!s}, out={n_outliers!s}')
    # plt.tight_layout(pad=2)

    ###
    # Display DBSCAN with only one eps value
    # print('len of projected_data =', len(projected_data))
    clustered = DBSCAN(eps=target_eps, min_samples=4).fit(projected_data)
    labels = clustered.labels_
    target_labels = labels
    # print('chksum target_labels =', sum(target_labels), 'len =', len(target_labels))
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = sum([(1 if label == -1 else 0) for label in labels])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    unique_colors = [plt.cm.get_cmap('viridis')(each) for each in np.linspace(0, 1, n_clusters)]

    # RGB
    unique_colors = [(1, 0, 0, 0),
                     (0, 1, 0, 0),
                     (0, 0, 1, 0),
                     (1, 1, 0, 0),
                     (1, 0, 1, 0),
                     (0, 1, 1, 0)]
    unique_colors = unique_colors[:n_clusters]
    unique_colors.append((0, 0, 0, 1))  # Black is the last color, for label -1

    colors = [unique_colors[label] for label in labels]

    plt.scatter(projected_data[:, 0], projected_data[:, 1],
                c=colors, alpha=0.5)


    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.title(f'eps={target_eps!s}, clu={n_clusters!s}, out={n_outliers!s}')
    plt.figlegend()
    plt.show()

    # print('chksum target_labels =', sum(target_labels))
    return target_labels, pca.components_