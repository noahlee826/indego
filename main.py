import ReadData as rd, matplotlib.pyplot as plt, numpy as np, k_means
from collections import Counter, defaultdict
from datetime import time

TRIPS_CSV_PATH = 'data/indego-trips-2018-q3.csv'
# TRIPS_CSV_PATH = 'data/indego-trips-2018-tiny.csv'
STATIONS_CSV_PATH = 'data/indego-stations-2018-10-19.csv'

VIRTUAL_STATION_STATION_ID = '3000'

# Morning rush: 6:30am - 9:30am
start_morning_window = time(6, 30, 0)
end_morning_window = time(9, 30, 0)

# Evening rush: 4:00pm - 7:00pm
start_evening_window = time(16, 0, 0)
end_evening_window = time(19, 0, 0)

trips = []
stations = []

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

rd.count_trips_per_station(trips, morning_outbound, morning_inbound,
                           start_morning_window, end_morning_window, weekdays_only=True)

rd.count_trips_per_station(trips, evening_outbound, evening_inbound,
                           start_evening_window, end_evening_window, weekdays_only=True)

# rd.count_trips_per_station_defaultdict(trips, morning_outbound_dd, morning_inbound_dd,
#                            start_morning_window, end_morning_window)
#
# rd.count_trips_per_station_defaultdict(trips, evening_outbound_dd, evening_inbound_dd,
#                            start_evening_window, end_evening_window)

print('Morning Outbound:', morning_outbound.most_common(10))
print('Morning Inbound: ', morning_inbound.most_common(10))
print('Evening Outbound:', evening_outbound.most_common(10))
print('Evening Inbound: ', evening_inbound.most_common(10))

station_counts = [{'station_id' : station['Station ID'],
                   'morning_out' : morning_outbound[station['Station ID']],
                   'morning_in' : morning_inbound[station['Station ID']],
                   'evening_out' : evening_outbound[station['Station ID']],
                   'evening_in' : evening_inbound[station['Station ID']]} for station in stations]

# rd.plot_station_counts(station_counts)

morning_data = list(zip([station['morning_out'] for station in station_counts],
                        [station['morning_in'] for station in station_counts]))
evening_data = list(zip([station['evening_out'] for station in station_counts],
                        [station['evening_in'] for station in station_counts]))

k_means.k_means_multiplot(morning_data, 4)
