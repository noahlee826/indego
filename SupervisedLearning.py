import ReadData as rd, GetData as gd
import numpy as np
from sklearn import linear_model, metrics
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
import random
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



# regr = linear_model.LinearRegression()
# print(X_train)
# print(y_train)
# regr.fit(X_train, y_train)
# linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)
# print(regr.coef_)
# print(regr.score(X_test, y_test))

solvers = ['lbfgs', 'liblinear']
num_trials = 10

confusion_matrices = defaultdict(list)
f1_scores = defaultdict(list)
coefficients = defaultdict(list)

features = [zg for zg in zoning_groups] + [CHAR_1, CHAR_2, CHAR_3]

for solver in solvers:
    print(solver)

    for i in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(scaled_data, targets, test_size=0.2)
        print('Bias in y_train:', sum(y_train), 'of', len(y_train), '(', float(sum(y_train) / len(y_train)), '%)')
        print('Bias in y_test :', sum(y_test), 'of', len(y_test), '(', float(sum(y_test) / len(y_test)), '%)')
        # print('Example rows from X_train:')
        # print(X_train[0])
        # print(X_train[1])
        # print(X_train[2])
        # print(X_train[3])
        # print()

        # Cross-Validation model
        clf = linear_model.LogisticRegressionCV(solver=solver, class_weight='balanced',
                                                n_jobs=-2, cv=5)
        clf.fit(X_train, y_train)
        expected = y_test

        predicted = clf.predict(X_test)

        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        confusion_matrices[solver] += [confusion_matrix(expected, predicted)]
        f1_scores[solver] += [f1_score(expected, predicted)]
        coefficients[solver] += [dict(zip(features, clf.coef_.tolist()[0]))]

        # fpr, tpr, threshhold = roc_curve(y_test, y_score)
        # roc_auc = auc(fpr, tpr)

        utils.print_hline()

        print('vvv Expected vvv')
        print(expected)
        print(predicted)
        print('^^^ Predicted ^^^')

        utils.print_hline()

        # print([zg for zg in zoning_groups] + [CHAR_1, CHAR_2, CHAR_3])
        # print(clf.coef_.tolist())
        coef_labels = zip(clf.coef_.tolist()[0], features)
        # coef_labels = zip(clf.coef_.tolist()[0], [zg for zg in zoning_groups] + [CHAR_1, CHAR_3])
        for label, coef in sorted(coef_labels, key=lambda x: abs(x[0]), reverse=True):
            print('{0:20}  {1}'.format(coef, label))

        utils.print_hline()

        print("Classification report for classifier %s:\n%s\n"
              % (clf, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

        utils.print_hline()

# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

utils.print_hline()
utils.print_hline()
utils.print_hline()

for solver in solvers:
    print(solver)
    print('Confusion Matrices:')
    for c_matrix in confusion_matrices[solver]:
        print(c_matrix)

    print()
    print('F1 Scores:')
    for f1_score in f1_scores[solver]:
        print(f1_score)
    print('Mean F1 score:\t', np.mean(f1_scores[solver]))

    print()
    print('Coefficients:')
    for coeffs in coefficients[solver]:
        for label, coef in sorted(coeffs.items(), key=utils.abs_idx1, reverse=True):
            print('{0:30}  {1}'.format(coef, label))
    print('Coefficient Averages:')
    avg_coeffs = {}
    for feature in features:
        cs = [coeffs[feature] for coeffs in coefficients[solver]]
        avg_coeffs[feature] = np.mean(cs)
    for item in sorted(avg_coeffs.items(), key=utils.abs_idx1, reverse=True):
        print(item)
    utils.print_hline()
