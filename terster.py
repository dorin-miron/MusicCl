import copy
from sklearn.externals import joblib

from utils import GENRE_LIST
from ceps import create_ceps_test, read_ceps_test

genre_list = GENRE_LIST

clf = None
clasificatori = []

def tst_model_on_single_file(file_path):
    for cc in range(3):
        print 'saved_model/model_ceps_%s.pkl' % str(cc)
        clasificatori.append(joblib.load('saved_model/model_ceps_%s.pkl' % str(cc)))

    clf_final = joblib.load('saved_model/model_ceps_f.pkl')

    X, y = read_ceps_test(create_ceps_test(test_file) + ".npy")
    X_GMM = []
    for cc in range(len(clasificatori)):
        clf = clasificatori[cc]
        proba = clf.predict_proba(X)
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

    probs = clf_final.predict_proba(X_GMM)
    print "\t".join(str(x) for x in traverse)
    print "\t".join(str("%.3f" % x) for x in probs[0])
    probs = probs[0]
    max_prob = max(probs)
    for i, j in enumerate(probs):
        if probs[i] == max_prob:
            max_prob_index = i

    print max_prob_index
    predicted_genre = traverse[max_prob_index]
    print "\n\npredicted genre = ", predicted_genre
    return predicted_genre


if __name__ == "__main__":

    global traverse
    traverse = GENRE_LIST

    test_file = "blues.00031.wav"
    predicted_genre = tst_model_on_single_file(test_file)
    print predicted_genre
