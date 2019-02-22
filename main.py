import ReadData as rd, matplotlib.pyplot as plt, numpy as np, k_means
from collections import Counter, defaultdict
from datetime import time

# TRIPS_CSV_PATH = 'data/indego-trips-2018-q3-10000.csv'
TRIPS_CSV_PATH = 'data/indego-trips-2018-q3.csv'
# TRIPS_CSV_PATH = 'data/indego-trips-2018-q3-tiny.csv'
STATIONS_CSV_PATH = 'data/indego-stations-2018-10-19.csv'

VIRTUAL_STATION_STATION_ID = '3000'


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

rd.extract_trips(TRIPS_CSV_PATH, trips)
rd.extract_stations(STATIONS_CSV_PATH, stations)
station_names = {station['Station ID']: station['Station Name'] for station in stations}

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

all_start_buckets, all_end_buckets, all_total_buckets = rd.count_bucketed_trips(trips, resolution=resolution, ok_days=days_to_analyze)
counts_by_stations = rd.count_bucketed_trips_by_station(trips, stations, resolution=resolution, ok_days=days_to_analyze)

all_start_relative = np.array([relative for absolute, relative in all_start_buckets.values()])
all_end_relative = np.array([relative for absolute, relative in all_end_buckets.values()])
all_total_relative = np.array([relative for absolute, relative in all_total_buckets.values()])


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
# rows_plots = (num_plots + 1) / 2  # Ensures the last plot is    n't cut off for an odd number of plots
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
# plt.figure(figsize=(10, 2))
# rd.plot_bucketed_count(all_total_buckets, use_relative_count=relative)
# plt.show()

######################################################################
# PCA
station_data = []
stn_ids = []
use_start_plus_end = True

for stn_id in station_names:
    stn_ids.append(stn_id)

    this_stn_data = []
    if use_start_plus_end:
        for count in counts_by_stations[stn_id][:2]:
            this_stn_data += [relative for absolute, relative in count.values()]
    else:
        count = counts_by_stations[stn_id][2]
        this_stn_data = np.array([relative for absolute, relative in count.values()])

    station_data.append(this_stn_data)
# projected_PCA_fit = rd.pca_dim_reduction(station_data)

#######################################
# DBSCAN
dbscan_labels, pca_components = rd.cluster_dbscan(station_data)
component_weights_1 = zip(all_total_buckets.keys(), pca_components[0])
component_weights_2 = zip(all_total_buckets.keys(), pca_components[1])

print('Component 1:')
for bucket, weight in sorted(component_weights_1, key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(bucket, weight)

print('\nComponent 2:')
for bucket, weight in sorted(component_weights_2, key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(bucket, weight)

print('chksum dbscan_labels =', sum(dbscan_labels), 'len =', len(dbscan_labels))
stn_labels = zip(stn_ids, dbscan_labels)
stn_labels = sorted(stn_labels, key=lambda x: x[1])

print('chksum stn_labels =', sum(label for _, label in stn_labels), 'len =', len(stn_labels))

for stn_id, label in stn_labels:
    print(label, station_names[stn_id])

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