import ReadData as rd, maya
from collections import Counter
from datetime import time

CSV_PATH = 'data/indego-trips-2018-q3.csv'
#CSV_PATH = 'data/indego-trips-2018-tiny.csv'

# Morning rush: 6:00am - 9:00am
start_morning_window = time(6, 0, 0)
end_morning_window = time(9, 0, 0)

# Evening rush: 3:30pm - 6:30pm
start_evening_window = time(15, 30, 0)
end_evening_window = time(18, 30, 0)

trips = []
outbound_station_counter = Counter()
inbound_station_counter = Counter()

rd.extract_trips(CSV_PATH, trips)

#rd.plot_trips(trips)

rd.count_trips_per_station(trips, outbound_station_counter, inbound_station_counter,
                           start_morning_window, end_morning_window)

print('Outbound:', outbound_station_counter.most_common(10))
print('Inbound: ', inbound_station_counter.most_common(10))
