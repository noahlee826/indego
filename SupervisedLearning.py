import ReadData as rd, GetData as gd
import numpy as np
from sklearn import linear_model, metrics
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import utils

import matplotlib.pyplot as plt

STATIONS_CSV_PATH = 'data/indego-stations-2018-10-19-with-labels.csv'

VIRTUAL_STATION_STATION_ID = '3000'
EXCLUDE_STATION_IDS = [VIRTUAL_STATION_STATION_ID]


extracted_stations = rd.extract_stations_from_file(STATIONS_CSV_PATH, station_list=None, exclude_station_ids=EXCLUDE_STATION_IDS)
stations_from_file = {stn['Station ID']: stn for stn in extracted_stations}

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

for stn in downloaded_station_chars:
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

print([zg for zg in zoning_groups] + [[CHAR_1], [CHAR_2], [CHAR_3]])
# print([zg for zg in zoning_groups] + [[CHAR_1], [CHAR_3]])

num_unbiased = 0
num_biased = 0
for stn_id, chars in sorted(list(stations_from_file.items())):
    # Don't use stations in our static file that we couldn't find online coords for
    # '2' is the label for bad data
    if stn_id in downloaded_station_ids and chars['Label'] != '2':
        # Next line is a kludgy manual integrity check that prints the station data
        # print(stn_id, chars['Label'], [chars[zg] for zg in zoning_groups] + [chars[CHAR_1], chars[CHAR_3]])
        data.append([chars[zg] for zg in zoning_groups] + [chars[CHAR_1], chars[CHAR_2], chars[CHAR_3]])
        # data.append([chars[zg] for zg in zoning_groups] + [chars[CHAR_1], chars[CHAR_3]])  #don't use CHAR_2
        targets.append(int(chars['Label']))
        print(stn_id, targets[-1], data[-1])
        if chars['Label'] == '0':
            num_unbiased += 1
        elif chars['Label'] == '1':
            num_biased += 1
print('Unbiased:', num_unbiased)
print('Biased:  ', num_biased)

num_continuous_features = 2
data = np.array(data)
targets = np.array(targets)

# Split into continuous and binary features so we can scale the continuous features without affecting the binary ones
continuous_data = data[:, -1 * num_continuous_features:]
binary_data = data[:, : -1 * num_continuous_features]
scaler = StandardScaler()
scaled_continuous_data = scaler.fit_transform(continuous_data)

scaled_data = np.concatenate((binary_data, scaled_continuous_data), axis=1)

X_train, X_test, y_train, y_test = train_test_split(scaled_data, targets, test_size=0.2)
print('Example rows from X_train:')
print(X_train[0])
print(X_train[1])
print(X_train[2])
print(X_train[3])
print()
print('Bias in y_train:', sum(y_train), 'of', len(y_train), '(', float(sum(y_train) / len(y_train)), '%)')
print('Bias in y_test :', sum(y_test), 'of', len(y_test), '(', float(sum(y_test) / len(y_test)), '%)')


# regr = linear_model.LinearRegression()
# print(X_train)
# print(y_train)
# regr.fit(X_train, y_train)
# linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)
# print(regr.coef_)
# print(regr.score(X_test, y_test))

# clf = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced')
clf = linear_model.LogisticRegressionCV(solver='lbfgs', class_weight='balanced', n_jobs=-2, cv=5) # Cross-Validation model
clf.fit(X_train, y_train)
expected = y_test

predicted = clf.predict(X_test)

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr, tpr, threshhold = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

utils.print_hline()

print('vvv Expected vvv')
print(expected)
print(predicted)
print('^^^ Predicted ^^^')

utils.print_hline()

# print([zg for zg in zoning_groups] + [CHAR_1, CHAR_2, CHAR_3])
# print(clf.coef_.tolist())
coef_labels = zip(clf.coef_.tolist()[0], [zg for zg in zoning_groups] + [CHAR_1, CHAR_2, CHAR_3])
# coef_labels = zip(clf.coef_.tolist()[0], [zg for zg in zoning_groups] + [CHAR_1, CHAR_3])
for coef, label in sorted(coef_labels, key=lambda x: abs(x[0]), reverse=True):
    print('{0:20}  {1}'.format(coef, label))

utils.print_hline()

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

utils.print_hline()

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()