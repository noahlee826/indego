import ReadData as rd, GetData as gd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics

STATIONS_CSV_PATH = 'data/indego-stations-2018-10-19-with-labels.csv'

VIRTUAL_STATION_STATION_ID = '3000'
EXCLUDE_STATION_IDS = [VIRTUAL_STATION_STATION_ID]


extracted_stations = rd.extract_stations_from_file(STATIONS_CSV_PATH, station_list=None, exclude_station_ids=EXCLUDE_STATION_IDS)
stations_from_file = {stn['Station ID'] : stn for stn in extracted_stations}

include_station_ids = set(stations_from_file.keys())
downloaded_station_zoning, zoning_groups = gd.get_zoning_data(include_station_ids=include_station_ids)
downloaded_station_chars = gd.get_station_characteristics(include_station_ids=include_station_ids)

downloaded_station_ids = set()
for stn in downloaded_station_zoning:
    stn_id = stn['station_id']
    downloaded_station_ids.add(stn_id)
    for zg in zoning_groups:
        # If the dl'ed station had a ZG, then the ZG=1 and others =0 because downloaded_station_zoning is a defaultdict
        # Eg record: {... 'Commercial/Commercial Mixed-Use': 1, 'Residential/Residential Mixed-Use': 0,...}
        stations_from_file[stn_id][zg] = stn[zg]

for stn in downloaded_station_zoning:
    stn_id = stn['station_id']
    if stn_id in include_station_ids:
        stations_from_file[stn_id]['pop_density'] = stn['pop_density']
        stations_from_file[stn_id]['pct_in_poverty'] = stn['pct_in_poverty']
        stations_from_file[stn_id]['med_income'] = stn['med_income']

# Properly assemble data
data = []
targets = []

CHAR_1 = 'pop_density'
CHAR_2 = 'med_income'
CHAR_3 = 'pct_in_poverty'

# for stn_id, chars in sorted(list(stations_from_file.items())):
#     # Don't use stations in our static file that we couldn't find online coords for
#     # '2' is the label for bad data
#     if stn_id in downloaded_station_ids and chars['Label'] != '2':
#         data.append([chars[CHAR_1], chars[CHAR_2], chars[CHAR_3]])
#         targets.append(int(chars['Label']))

print([zg for zg in zoning_groups])
for stn_id, chars in sorted(list(stations_from_file.items())):
    # Don't use stations in our static file that we couldn't find online coords for
    # '2' is the label for bad data
    if stn_id in downloaded_station_ids and chars['Label'] != '2':
        # Next line is a kludgy manual integrity check that prints a roughly random subset of station's data
        print(stn_id, [chars[zg] for zg in zoning_groups] + [CHAR_1, CHAR_2, CHAR_3])
        data.append([chars[zg] for zg in zoning_groups])
        targets.append(int(chars['Label']))

data = np.array(data)
targets = np.array(targets)

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)

# regr = linear_model.LinearRegression()
# print(X_train)
# print(y_train)
# regr.fit(X_train, y_train)
# linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)
# print(regr.coef_)
# print(regr.score(X_test, y_test))

clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
expected = y_test
predicted = clf.predict(X_test)

labels = ['0 No bias', '1 Bias']

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))