import copy
from sklearn.externals import joblib

from utils import GENRE_LIST
from utils import TEST_DIR
from ceps import create_ceps_test, read_ceps_test

genre_list = GENRE_LIST
dir_test = TEST_DIR

clf = None
classifiers = []


def tst_model_on_single_file(test_file):
    for cc in range(6):
        print 'saved_model/model_ceps_%s.pkl' % str(cc)
        classifiers.append(joblib.load('saved_model/model_ceps_%s.pkl' % str(cc)))

    clf_final = joblib.load('saved_model/model_ceps_f.pkl')

    input_x, y = read_ceps_test(create_ceps_test(test_file) + ".npy")
    computed_x = []
    for cc in range(len(classifiers)):
        clf = classifiers[cc]
        proba = clf.predict_proba(input_x)
        if len(computed_x) == 0:
            computed_x = copy.copy(proba)
        else:
            aux = [r for r in zip(computed_x, proba)]
            computed_x = []
            for x in aux:
                computed_x.append([])
                for y in x:
                    for z in y:
                        computed_x[-1].append(z)

    probs = clf_final.predict_proba(computed_x)
    print "\t".join(str(x) for x in GENRE_LIST)
    print "\t".join(str("%.3f" % x) for x in probs[0])
    probs = probs[0]
    max_prob = max(probs)
    max_prob_index = 0
    for i, j in enumerate(probs):
        if probs[i] == max_prob:
            max_prob_index = i

    print max_prob_index
    predicted_genre = GENRE_LIST[max_prob_index]
    print "\n\npredicted genre = ", predicted_genre
    return predicted_genre


if __name__ == "__main__":

    file_name = "classical.00006.wav"
    tst_model_on_single_file(dir_test+"/"+file_name)
