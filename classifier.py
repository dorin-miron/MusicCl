import os
import copy
import itertools
import timeit
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from utils import GENRE_LIST, GENRE_DIR, TEST_DIR
from utils import plot_confusion_matrix, plot_roc_curves
from ceps import read_ceps, read_ceps_test

from sklearn.model_selection import ShuffleSplit as ShuffleSp

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn import mixture

from sklearn.neural_network import MLPClassifier

genre_list = GENRE_LIST
original_params = {'n_estimators': 100, 'max_leaf_nodes': 16, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}
clasificatori = [LogisticRegression(),
                RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=1),
                GradientBoostingClassifier(**original_params)]

#clasificatori = [MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)]


def train_model(X, Y, name, plot=False):
    """
        train_model(vector, vector, name[, plot=False])
        
        Trains and saves model to disk.
    """
    labels = np.unique(Y)

    cv = ShuffleSplit(n=len(X), test_size=0.4, random_state=1)

    train_errors = []
    test_errors = []

    scores = []
    pr_scores = defaultdict(list)
    precisions, recalls, thresholds = defaultdict(list), defaultdict(list), defaultdict(list)

    roc_scores = defaultdict(list)
    tprs = defaultdict(list)
    fprs = defaultdict(list)

    clfs = [[] for _ in range(len(clasificatori))] # for the median

    cms = [[] for _ in range(len(clasificatori))]

    #GMM_clf = mixture.GaussianMixture(n_components=10, covariance_type='full',max_iter=300)
    GMM_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=1)


    X_GMM_F = []
    Y_GMM = []
    thefile = open('test.txt', 'w')

    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

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
                plot_roc_curves(roc_scores[label][median], desc, tprs[label][median],fprs[label][median], label='%s vs rest' % genre_list[label])

        joblib.dump(clasificatori[cc], 'saved_model/model_ceps_%s.pkl' % str(cc))

    all_pr_scores = np.asarray(pr_scores.values()).flatten()
    summary = (np.mean(scores), np.std(scores), np.mean(all_pr_scores), np.std(all_pr_scores))
    #print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)


    #save the trained model to disk
    print "lungime input"
    print len(X_GMM_F)

    print "lungime output"
    print len(list(Y_GMM))

    cv_gmm = ShuffleSp(n_splits=10, test_size=0.4, random_state=1)
    gmm_cms = []

    for train, test in cv_gmm.split(X_GMM_F):
        X_train = []
        y_train = []
        for el in train:
            X_train.append(X_GMM_F[el])
            y_train.append(Y_GMM[el])

        X_test = []
        y_test = []

        for el in test:
            X_test.append(X_GMM_F[el])
            y_test.append(Y_GMM[el])

        GMM_clf.fit(X_train, y_train)

        y_pred = GMM_clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        gmm_cms.append(cm)


    print "plot EM confusion matrix"

    cm_avg = np.mean(gmm_cms, axis=0)
    cm_norm = cm_avg / np.sum(cm_avg, axis=0)
    print cm_norm
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
        plot_confusion_matrix(cm_norm, genre_list, "ceps"+str(cc), "CEPS classifier - Confusion matrix")

import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


if __name__ == "__main__":
    start = timeit.default_timer()
    print
    print " Starting classification \n"
    print " Classification running ... \n" 
    X, y = read_ceps(genre_list)
    train_avg, test_avg = train_model(X, y, "ceps", plot=True)
    print " Classification finished \n"
    stop = timeit.default_timer()
    print " Total time taken (s) = ", (stop - start)
    print " All Done\n"
    print " See plots in 'graphs' directory \n"
    
