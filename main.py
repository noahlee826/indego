import os, ReadData as rd, matplotlib.pyplot as plt, numpy as np, utils
from collections import Counter, defaultdict
from datetime import time

TRIPS_CSV_PATH = 'data/indego-trips-2018-q3.csv'
# TRIPS_CSV_PATH = 'data/indego-trips-2018-q3-10000.csv'
# TRIPS_CSV_PATH = 'data/indego-trips-2018-q3-tiny.csv'
TRIPS_CSV_DIR = 'data//2018-all'


# STATIONS_CSV_PATH = 'data/indego-stations-2018-10-19.csv'
STATIONS_CSV_PATH = 'data/indego-stations-2018-10-19-with-labels.csv'

VIRTUAL_STATION_STATION_ID = '3000'
EXCLUDE_STATION_IDS = [VIRTUAL_STATION_STATION_ID]

FIG_FOLDER_PATH = "C:/Users/Noah/Google Drive/@@_SuperSenior/Plots/picdump/"
FIG_FORMAT = 'png'

# Peak hours as defined by SEPTA's bike-on-subway policy: http://www.septa.org/policy/bike.html
# Morning rush: 6:00am - 9:00am
start_morning_window = time(6, 0, 0)
end_morning_window = time(9, 0, 0)

# Evening rush: 3:00pm - 6:00pm
start_evening_window = time(15, 0, 0)
end_evening_window = time(18, 0, 0)

trips = []
stations = []

weekdays = [1, 2, 3, 4, 5]
weekends = [6, 7]
all_days = [1, 2, 3, 4, 5, 6, 7]

morning_outbound = Counter()
morning_inbound = Counter()
evening_outbound = Counter()
evening_inbound = Counter()

morning_outbound_dd = defaultdict(int)
morning_inbound_dd = defaultdict(int)
evening_outbound_dd = defaultdict(int)
evening_inbound_dd = defaultdict(int)

# Exclude rows where 'Virtual Station' is the beginning or end
stations = rd.extract_stations_from_file(STATIONS_CSV_PATH, stations, EXCLUDE_STATION_IDS)
station_names = {station['Station ID']: station['Station Name'] for station in stations}
station_names = defaultdict(lambda: 'Station Name Missing', station_names)
# trips = rd.extract_trips(TRIPS_CSV_PATH, trips)
trips = rd.extract_trips_dir(TRIPS_CSV_DIR, trips, EXCLUDE_STATION_IDS)


# Convenience mapping to easily find Station Name from Station ID

# rd.count_trips_per_station(trips, morning_outbound, morning_inbound,
#                            start_morning_window, end_morning_window, ok_days=weekdays)
#
# rd.count_trips_per_station(trips, evening_outbound, evening_inbound,
#                            start_evening_window, end_evening_window, ok_days=weekdays)

# rd.count_trips_per_station_defaultdict(trips, morning_outbound_dd, morning_inbound_dd,
#                            start_morning_window, end_morning_window)
#
# rd.count_trips_per_station_defaultdict(trips, evening_outbound_dd, evening_inbound_dd,
#                            start_evening_window, end_evening_window)

# print('Morning Outbound:', morning_outbound.most_common(10))
# print('Morning Inbound: ', morning_inbound.most_common(10))
# print('Evening Outbound:', evening_outbound.most_common(10))
# print('Evening Inbound: ', evening_inbound.most_common(10))

# station_counts = [{'station_id' : station['Station ID'],
#                    'morning_out' : morning_outbound[station['Station ID']],
#                    'morning_in' : morning_inbound[station['Station ID']],
#                    'evening_out' : evening_outbound[station['Station ID']],
#                    'evening_in' : evening_inbound[station['Station ID']],
#                    'morning_outin_ratio' : morning_outbound[station['Station ID']] / morning_inbound[station['Station ID']],
#                    'evening_outin_ratio' : evening_outbound[station['Station ID']] / evening_inbound[station['Station ID']]
#                     } for station in stations if morning_inbound[station['Station ID']] != 0 and evening_inbound[station['Station ID']]]

# rd.plot_station_counts(station_counts)

# morning_data = list(zip([station['morning_out'] for station in station_counts],
#                         [station['morning_in'] for station in station_counts]))
# evening_data = list(zip([station['evening_out'] for station in station_counts],
#                         [station['evening_in'] for station in station_counts]))
#
# ratio_data = list(zip([station['morning_outin_ratio'] for station in station_counts],
#                       [station['evening_outin_ratio'] for station in station_counts]))

# k_means.k_means_multiplot(morning_data, 4, xlabel='AM Out', ylabel='AM In')
# k_means.k_means_multiplot(evening_data, 4, xlabel='PM Out', ylabel='PM In')
# k_means.k_means_multiplot(ratio_data, 4, xlabel='AM Out : AM In', ylabel='PM Out : PM In')

days_to_analyze = all_days
resolution = 15

print('Counting bucketed trips system-wide...')
all_start_buckets, all_end_buckets, all_total_buckets = rd.count_bucketed_trips(trips,
                                                                                resolution=resolution,
                                                                                ok_days=days_to_analyze)

print('Counting bucketed trips per station...')
counts_by_stations = rd.count_bucketed_trips_by_station(trips, stations,
                                                        resolution=resolution,
                                                        ok_days=days_to_analyze)

total_for_all_stns = sum([absolute for absolute, relative in all_total_buckets.values()])

all_start_relative = np.array([relative for absolute, relative in all_start_buckets.values()])
all_end_relative = np.array([relative for absolute, relative in all_end_buckets.values()])
all_total_relative = np.array([relative for absolute, relative in all_total_buckets.values()])

bucket_texts = all_total_buckets.keys()

# For each station, find the deviance from the mean
# and save the plot in FIG_FOLDER_PATH
for stn_id, counts_by_station in counts_by_stations.items():
    # print('Working on station', station['Station ID'], station['Station Name'])
    # print('Working on station', stn_id, station_names[stn_id])

    # stn_id = station['Station ID']

    # buckets = counts_by_stations[stn_id][0] # 0 = trip starts, 1 = trip ends, 2 = total
    buckets = counts_by_station[0]  # 0 = trip starts, 1 = trip ends, 2 = total

    # print(buckets)

    actual_counts = [list(abs_rel_tuple) for abs_rel_tuple in zip(*buckets.values())]
    overall_counts = [list(abs_rel_tuple) for abs_rel_tuple in zip(*all_total_buckets.values())]

    # Calculate difference for absolute counts
    actual_absolute_counts = actual_counts[0]
    overall_absolute_counts = overall_counts[0]
    # Need to prorate the overall absolute counts down to what we'd expect those counts to be based on this station's
    # portion of the overall total rides
    total_for_this_stn = sum(actual_absolute_counts)
    stn_percent = total_for_this_stn / total_for_all_stns
    expected_absolute_counts = [stn_percent * absolute for absolute in overall_absolute_counts]
    absolute_diffs = [actual - expected for actual, expected in zip(actual_absolute_counts, expected_absolute_counts)]

    # Calculate difference for relative counts
    actual_relative_counts = actual_counts[1]
    # Relative counts don't need "scaling down", since they're already relative
    expected_relative_counts = overall_counts[1]
    relative_diffs = [actual - expected for actual, expected in zip(actual_relative_counts, expected_relative_counts)]

    stn_total_buckets = dict(zip(bucket_texts, list(zip(absolute_diffs, relative_diffs))))

    use_relative = True

    plt.figure(figsize=(10, 3))
    rd.plot_bucketed_count(stn_total_buckets, use_relative_count=use_relative)
    clean_stn_name = station_names[stn_id].replace('"', '')
    plt.title(clean_stn_name)
    path = FIG_FOLDER_PATH + stn_id + ' ' + clean_stn_name + '.' + FIG_FORMAT
    plt.xlabel('Time of Day')
    plt.ylabel('Portion of Avg Daily Trips')
    if use_relative:
        plt.ylim([-0.06, 0.08])
        plt.yticks(np.arange(-0.06, 0.08, 0.01))
    plt.grid(True, axis='y', which='both')
    plt.tight_layout()
    # plt.show()
    # print('Saving plot to:', path)
    plt.savefig(path, format=FIG_FORMAT)
    plt.close()

########################################
# MSE Analysis
# start_MSEs = {}
# end_MSEs = {}
# total_MSEs = {}
#
# for station in stations:
#     stn_id = station['Station ID']
#     buckets = counts_by_stations[stn_id]
#     if (sum([absolute for absolute, _ in buckets[2].values()])) > 0.00 * len(trips):
#         stn_start_relative = np.array([relative for absolute, relative in buckets[0].values()])
#         stn_end_relative = np.array([relative for absolute, relative in buckets[1].values()])
#         stn_total_relative = np.array([relative for absolute, relative in buckets[2].values()])
#
#         start_MSEs[stn_id] = rd.mse(stn_start_relative, all_start_relative)
#         end_MSEs[stn_id] = rd.mse(stn_end_relative, all_end_relative)
#         total_MSEs[stn_id] = rd.mse(stn_total_relative, all_total_relative)
#
# max_start_MSE_stn, max_start_MSE = max(start_MSEs.items(), key=lambda k: k[1])
# print(max_start_MSE_stn, station_names[max_start_MSE_stn], max_start_MSE)
#
# plt.figure(figsize=(10, 10))
# num_plots = 10
# rows_plots = (num_plots + 1) / 2  # Ensures the last plot isn't cut off for an odd number of plots
# cols_plots = 2
# plot_ind = 0
#
# relative = True
#
# for stn_id, mse in sorted(total_MSEs.items(), key=lambda x: x[1], reverse=True):
#     if plot_ind < 10:
#         plot_ind += 1
#         plt.subplot(rows_plots, cols_plots, plot_ind)
#         rd.plot_bucketed_count(counts_by_stations[stn_id][2], use_relative_count=relative, group_n=0)
#         plt.title(f'{station_names[stn_id]}, MSE = {mse!s}')
#
# plt.tight_layout()
# plt.show()

######################################################################
# Total bucket plot
plt.figure(figsize=(10, 2))
rd.plot_bucketed_count(all_total_buckets, use_relative_count=True)
plt.show()

######################################################################
# Preprocess data for DBSCAN
stn_ids = []
for stn_id in station_names:
    stn_ids.append(stn_id)

station_data = rd.assemble_station_data(stn_ids, counts_by_stations)

######################################################################
# DBSCAN
dbscan_labels, pca_components = rd.cluster_dbscan(station_data)
component_weights_1 = zip(all_total_buckets.keys(), pca_components[0])
component_weights_2 = zip(all_total_buckets.keys(), pca_components[1])

######################################################################
# Print PCA component weights
# print('Component 1:')
# for bucket, weight in sorted(component_weights_1, key=utils.abs_idx1, reverse=True)[:10]:
#     print(bucket, weight)
# print()
# print('Component 2:')
# for bucket, weight in sorted(component_weights_2, key=utils.abs_idx1, reverse=True)[:10]:
#     print(bucket, weight)

# Associate each station ID with its DBSCAN label
stn_labels = zip(stn_ids, dbscan_labels)
# Sort by label
stn_labels = sorted(stn_labels, key=utils.idx1)
#Print the label
# for stn_id, label in stn_labels:
#     print(label, station_names[stn_id])

# test_stn_id = max_start_MSE_stn
#
# start_buckets = counts_by_stations[test_stn_id][0]
# end_buckets = counts_by_stations[test_stn_id][1]
# total_buckets = counts_by_stations[test_stn_id][2]
#
#
# plt.figure(figsize=(10, 12))
# plt.subplot(311)
# rd.plot_bucketed_count(start_buckets, use_relative_count=relative)
# plt.ylabel('Trip Start Volume')
# plt.subplot(312)
# rd.plot_bucketed_count(end_buckets, use_relative_count=relative)
# plt.ylabel('Trip End Volume')
# plt.subplot(313)
# rd.plot_bucketed_count(total_buckets, use_relative_count=relative)
# plt.ylabel('Total Trip Volume')

# plt.tight_layout(pad=2)
# plt.show()