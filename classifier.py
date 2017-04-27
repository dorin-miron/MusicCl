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
from sklearn.metrics import accuracy_score

from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from sklearn.decomposition import PCA

genre_list = GENRE_LIST
first_plot = True

original_params = {'n_estimators': 100, 'max_leaf_nodes': 8, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}

classifiers = [MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1),
               LogisticRegression(),
               SVC(kernel='linear', probability=True),
               RandomForestClassifier(n_estimators=100, max_depth=None, random_state=1),
               GradientBoostingClassifier(**original_params),
               QuadraticDiscriminantAnalysis(),
               KNeighborsClassifier(5)]

# GMM_clf = mixture.GaussianMixture(n_components=10, covariance_type='full',max_iter=300)
GMM_clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=2)


def plot_decision(real_x, real_y, pred_y, name):
    # TODO see where is commented and modify to display x and real y from test_dataset
    global first_plot
    background_model = None

    x_train_embedded = PCA(n_components=2).fit_transform(real_x)

    resolution = 1000  # 100x100 background pixels
    x2d_x_min, x2d_x_max = np.min(x_train_embedded[:, 0]), np.max(x_train_embedded[:, 0])
    x2d_y_min, x2d_y_max = np.min(x_train_embedded[:, 1]), np.max(x_train_embedded[:, 1])
    xx, yy = np.meshgrid(np.linspace(x2d_x_min, x2d_x_max, resolution), np.linspace(x2d_y_min, x2d_y_max, resolution))

    background_model = KNeighborsClassifier(n_neighbors=1).fit(x_train_embedded, pred_y)
    voronoi_background = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoi_background = voronoi_background.reshape((resolution, resolution))

    # plot
    # voronoi_background[-1][-1] = 10
    plt.contourf(xx, yy, voronoi_background, 10)
    # plt.scatter(X_Train_embedded[:, 0], x_train_embedded[:, 1], c=real_y)
    if first_plot:
        cbar = plt.colorbar()
        cbar.ax.set_yticklabels(GENRE_LIST)
        first_plot = False
    plt.savefig(os.path.join(CHART_DIR, name), bbox_inches="tight")


def train_model(real_x, real_y, name, plot=False):
    labels = np.unique(real_y)

    cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)

    train_errors = []
    test_errors = []

    scores = []
    # pr_scores = defaultdict(list)
    # precisions, recalls, thresholds = defaultdict(list), defaultdict(list), defaultdict(list)

    roc_scores = defaultdict(list)
    tprs = defaultdict(list)
    fprs = defaultdict(list)

    clfs = [[] for _ in range(len(classifiers))]  # for the median

    cms = [[] for _ in range(len(classifiers))]

    computed_x = []
    computed_y = []

    plot_x = []
    plot_y = []
    acc_test_x = []
    acc_test_y = []

    for train, test in cv.split(real_x):
        x_train, y_train = extract_sample(real_x, real_y, train)
        x_test, y_test = extract_sample(real_x, real_y, test)

        plot_x = list(itertools.chain(plot_x, x_train))
        plot_y = list(itertools.chain(plot_y, y_train))

        acc_test_x = list(itertools.chain(acc_test_x, x_test))
        acc_test_y = list(itertools.chain(acc_test_y, y_test))

        intermediate_x = []

        for cc in range(len(classifiers)):
            clf = classifiers[cc]

            clf.fit(x_train, y_train)
            clfs[cc].append(clf)

            train_score = clf.score(x_train, y_train)
            test_score = clf.score(x_test, y_test)
            scores.append(test_score)

            train_errors.append(1 - train_score)
            test_errors.append(1 - test_score)

            y_pred = clf.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)
            cms[cc].append(cm)

            for label in labels:
                y_label_test = np.asarray(y_test == label, dtype=int)
                proba = clf.predict_proba(x_test)
                proba_label = proba[:, label]

                fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
                roc_scores[label].append(auc(fpr, tpr))
                tprs[label].append(tpr)
                fprs[label].append(fpr)

            proba = clf.predict_proba(x_test)
            if len(intermediate_x) == 0:
                intermediate_x = copy.copy(proba)
            else:
                aux = [r for r in zip(intermediate_x, proba)]
                intermediate_x = []
                for x in aux:
                    intermediate_x.append([])
                    for k in x:
                        for z in k:
                            intermediate_x[-1].append(z)

        computed_y = list(itertools.chain(computed_y, y_test))
        if len(computed_x) == 0:
            computed_x = copy.copy(intermediate_x)
        else:
            computed_x = computed_x + intermediate_x

    for cc in range(len(classifiers)):
        if plot:
            for label in labels:
                scores_to_sort = roc_scores[label]
                median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
                desc = "%s_%s %s" % (name, cc, genre_list[label])
                plot_roc_curves(roc_scores[label][median], desc, tprs[label][median], fprs[label][median],
                                label='%s vs rest' % genre_list[label])

        joblib.dump(classifiers[cc], 'saved_model/model_ceps_%s.pkl' % str(cc))
        #plot_decision(plot_x, plot_y, classifiers[cc].predict(plot_x), "clasificatorul_%s" %str(cc))
        print "acuratetea %s este: " % cc, accuracy_score(acc_test_y, classifiers[cc].predict(acc_test_x))

    # all_pr_scores = np.asarray(pr_scores.values()).flatten()
    # summary = (np.mean(scores), np.std(scores), np.mean(all_pr_scores), np.std(all_pr_scores))
    # print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

    # save the trained model to disk
    cv_gmm = ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)
    gmm_cms = []
    all_train_x = []
    all_train_y = []
    y_predicted = []

    acc_final_x = []
    acc_final_y = []

    for train, test in cv_gmm.split(computed_x):
        x_train, y_train = extract_sample(computed_x, computed_y, train)
        x_test, y_test = extract_sample(computed_x, computed_y, test)

        acc_final_x = list(itertools.chain(acc_final_x, x_test))
        acc_final_y = list(itertools.chain(acc_final_y, y_test))

        GMM_clf.fit(x_train, y_train)

        y_pred = GMM_clf.predict(x_test)

        if len(all_train_x) == 0:
            all_train_x = x_train
            all_train_y = y_train
        else:
            all_train_x = all_train_x + x_train
            all_train_y = all_train_y + y_train

        for r in GMM_clf.predict(x_train):
            y_predicted.append(r)

        cm = confusion_matrix(y_test, y_pred)
        gmm_cms.append(cm)

    plot_decision(all_train_x, all_train_y, y_predicted, "final_classifier")
    print "Acuratetea clasificatorului final este: ", accuracy_score(acc_final_y, GMM_clf.predict(acc_final_x))

    print "plot EM confusion matrix"

    cm_avg = np.mean(gmm_cms, axis=0)
    cm_norm = cm_avg / np.sum(cm_avg, axis=0)
    print "\n Plotting confusion matrix ... \n"
    plot_confusion_matrix(cm_norm, genre_list, "Final_Classifier", "CEPS classifier - Confusion matrix")

    joblib.dump(GMM_clf, 'saved_model/model_ceps_f.pkl')
    conf_matrix(np.asarray(cms))

    return np.mean(train_errors), np.mean(test_errors)


def conf_matrix(cms):
    for cc in range(len(classifiers)):
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
