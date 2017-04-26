import os
import copy
import timeit
import itertools
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from utils import GENRE_LIST, CHART_DIR
from utils import plot_confusion_matrix, plot_roc_curves, extract_sample
from sklearn.neighbors.classification import KNeighborsClassifier
from ceps import read_ceps

from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.decomposition import PCA
from matplotlib import ticker

genre_list = GENRE_LIST

original_params = {'n_estimators': 100, 'max_leaf_nodes': 16, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}

clasificatori = [LogisticRegression(),
                 RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=1),
                 GradientBoostingClassifier(**original_params)]

# GMM_clf = mixture.GaussianMixture(n_components=10, covariance_type='full',max_iter=300)
GMM_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)


def plot_decision(real_x, real_y, pred_y):
    #TODO see where is commented and modify to display x and real y from test_dataset

    x_train_embedded = PCA(n_components=2).fit_transform(real_x)

    resolution = 1000  # 100x100 background pixels
    x2d_xmin, x2d_xmax = np.min(x_train_embedded[:, 0]), np.max(x_train_embedded[:, 0])
    x2d_ymin, x2d_ymax = np.min(x_train_embedded[:, 1]), np.max(x_train_embedded[:, 1])
    xx, yy = np.meshgrid(np.linspace(x2d_xmin, x2d_xmax, resolution), np.linspace(x2d_ymin, x2d_ymax, resolution))

    background_model = KNeighborsClassifier(n_neighbors=1).fit(x_train_embedded, pred_y)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    # plot
    # voronoiBackground[-1][-1] = 10
    plt.contourf(xx, yy, voronoiBackground, 10)
    # plt.scatter(X_Train_embedded[:, 0], x_train_embedded[:, 1], c=real_y)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(GENRE_LIST)
    plt.savefig(os.path.join(CHART_DIR, "test"), bbox_inches="tight")


def train_model(X, Y, name, plot=False):
    labels = np.unique(Y)

    cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)

    train_errors = []
    test_errors = []

    scores = []
    pr_scores = defaultdict(list)
    precisions, recalls, thresholds = defaultdict(list), defaultdict(list), defaultdict(list)

    roc_scores = defaultdict(list)
    tprs = defaultdict(list)
    fprs = defaultdict(list)

    clfs = [[] for _ in range(len(clasificatori))]  # for the median

    cms = [[] for _ in range(len(clasificatori))]

    X_GMM_F = []
    Y_GMM = []

    for train, test in cv.split(X):
        X_train, y_train = extract_sample(X, Y, train)
        X_test, y_test = extract_sample(X, Y, test)

        X_GMM = []

        for cc in range(len(clasificatori)):
            clf = clasificatori[cc]

            clf.fit(X_train, y_train)
            clfs[cc].append(clf)

            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            scores.append(test_score)

            train_errors.append(1 - train_score)
            test_errors.append(1 - test_score)

            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cms[cc].append(cm)

            for label in labels:
                y_label_test = np.asarray(y_test == label, dtype=int)
                proba = clf.predict_proba(X_test)
                proba_label = proba[:, label]

                fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
                roc_scores[label].append(auc(fpr, tpr))
                tprs[label].append(tpr)
                fprs[label].append(fpr)

            proba = clf.predict_proba(X_test)
            if len(X_GMM) == 0:
                X_GMM = copy.copy(proba)
            else:
                aux = [r for r in zip(X_GMM, proba)]
                X_GMM = []
                for x in aux:
                    X_GMM.append([])
                    for y in x:
                        for z in y:
                            X_GMM[-1].append(z)

        Y_GMM = list(itertools.chain(Y_GMM, y_test))
        if len(X_GMM_F) == 0:
            X_GMM_F = X_GMM
        else:
            X_GMM_F = X_GMM_F + X_GMM

    for cc in range(len(clasificatori)):
        if plot:
            for label in labels:
                scores_to_sort = roc_scores[label]
                median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
                desc = "%s_%s %s" % (name, cc, genre_list[label])
                plot_roc_curves(roc_scores[label][median], desc, tprs[label][median], fprs[label][median],label='%s vs rest' % genre_list[label])

        joblib.dump(clasificatori[cc], 'saved_model/model_ceps_%s.pkl' % str(cc))

    # all_pr_scores = np.asarray(pr_scores.values()).flatten()
    # summary = (np.mean(scores), np.std(scores), np.mean(all_pr_scores), np.std(all_pr_scores))
    # print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

    # save the trained model to disk
    cv_gmm = ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)
    gmm_cms = []
    all_train_x = []
    all_train_y = []
    y_predicted = []

    for train, test in cv_gmm.split(X_GMM_F):
        X_train, y_train = extract_sample(X_GMM_F, Y_GMM, train)
        X_test, y_test = extract_sample(X_GMM_F, Y_GMM, test)

        GMM_clf.fit(X_train, y_train)

        y_pred = GMM_clf.predict(X_test)

        if len(all_train_x) == 0:
            all_train_x = X_train
            all_train_y = y_train
        else:
            all_train_x = all_train_x + X_train
            all_train_y = all_train_y + y_train

        for r in GMM_clf.predict(X_train):
            y_predicted.append(r)

        cm = confusion_matrix(y_test, y_pred)
        gmm_cms.append(cm)

    plot_decision(all_train_x, all_train_y, y_predicted)

    print "plot EM confusion matrix"

    cm_avg = np.mean(gmm_cms, axis=0)
    cm_norm = cm_avg / np.sum(cm_avg, axis=0)
    print "\n Plotting confusion matrix ... \n"
    plot_confusion_matrix(cm_norm, genre_list, "EM", "CEPS classifier - Confusion matrix")

    joblib.dump(GMM_clf, 'saved_model/model_ceps_f.pkl')
    conf_matrix(np.asarray(cms))

    return np.mean(train_errors), np.mean(test_errors)


def conf_matrix(cms):
    for cc in range(len(clasificatori)):
        cm_avg = np.mean(cms[cc], axis=0)
        cm_norm = cm_avg / np.sum(cm_avg, axis=0)
        print "\n Plotting confusion matrix ... \n"
        plot_confusion_matrix(cm_norm, genre_list, "ceps" + str(cc), "CEPS classifier - Confusion matrix")

if __name__ == "__main__":
    start = timeit.default_timer()
    print
    print " Starting classification \n"
    print " Classification running ... \n"
    X, y = read_ceps(genre_list)

    # TODO change plot to True when I want to create roc figures
    train_avg, test_avg = train_model(X, y, "ceps", plot=False)

    print " Classification finished \n"
    stop = timeit.default_timer()
    print " Total time taken (s) = ", (stop - start)
    print " All Done\n"
    print " See plots in 'graphs' directory \n"
