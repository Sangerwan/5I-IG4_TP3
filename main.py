import cv2
import numpy as np
import glob
import sklearn
import sklearn.cluster
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import dml

training_folders = ["Img/pigeon/*.jpg", "Img/pizza/*.jpg", "Img/scorpion/*.jpg", "Img/sunflower/*.jpg"]
test_folders = ["Img/pigeon/test/*.jpg", "Img/pizza/test/*.jpg", "Img/scorpion/test/*.jpg", "Img/sunflower/test/*.jpg"]

def vocabulaire(N, chemins, fichier=None, methode = "kmeans"):
    s=cv2.SIFT_create()
    listdesc = []
    for chemin in chemins:
        for image in glob.glob(chemin):
            #print(image)
            img=cv2.imread(image)
            kpts, descriptors = s.detectAndCompute(img, None)
            for desc in descriptors:
                listdesc.append(desc)

    if methode == "kmeans":
        array = np.array(listdesc)
        kmeans =  sklearn.cluster.KMeans(N)
        predict = kmeans.fit_predict(array)
        max = 0
        for i in range(len(predict)):
            dist = np.linalg.norm(kmeans.cluster_centers_[predict[i]]-array[i])
            if dist > max:
                max = dist

    if fichier is not None:
        np.savetxt(fichier, kmeans.cluster_centers_, delimiter=',')

    return kmeans.inertia_/N, max

def coude():
    #N= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    N= range(1,100)
    inertie = []
    max = []
    for n in N:
        print("----- %s -----" % n)
        chemins = training_folders
        inertia, max_dist = vocabulaire(n, chemins)
        inertie.append(inertia)
        max.append(max_dist)
    plt.figure(1)
    plt.plot(N, inertie)    
    plt.figure(2)
    plt.plot(N, max)
    plt.show()

def vectoriser(im, voca):
    vocabulaire = np.loadtxt(voca, delimiter=',')
    img=cv2.imread(im)
    s=cv2.SIFT_create()
    kpt, descriptor = s.detectAndCompute(img, None)
    hist = np.zeros(len(vocabulaire))
    for desc in descriptor:
        index = np.argmin(np.linalg.norm(vocabulaire-desc, axis=1))
        hist[index] += 1
    return hist

def test_veco():
    dict = {}
    for chemin in training_folders:
        list = []
        image_class = chemin.split('/')[1]
        for image in glob.glob(chemin):
            hist = vectoriser(image, "voca15.txt")
            list.append(hist)
        dict[image_class] = list
    with open("base.pickle", "wb") as f:
        pickle.dump(dict, f)

def test_veco_training():
    dict = {}
    for chemin in test_folders:
        list = []
        image_class = chemin.split('/')[1]
        for image in glob.glob(chemin):
            hist = vectoriser(image, "voca15.txt")
            list.append(hist)
        dict[image_class] = list
    with open("test.pickle", "wb") as f:
        pickle.dump(dict, f)

def get_data():
    with open("base.pickle", "rb") as f:
        data = pickle.load(f)
    with open("test.pickle", "rb") as f:
        test = pickle.load(f)
    return data, test


# chemins = ["Img/pigeon/test/*.jpg", "Img/pizza/test/*.jpg", "Img/scorpion/test/*.jpg", "Img/sunflower/test/*.jpg"]
# inertia, max_dist = vocabulaire(10, chemins)

#vocabulaire(15, training_folders, "voca15.txt")
#coude()
#vectoriser("Img/pigeon/test/image_0015.jpg", "voca10.txt")
# test_veco()
# test_veco_training()
"""
[[1.76144233e+07]
 [1.27533930e+06]
 [5.52572461e+06]
 [4.66433866e+07]
 [1.82054071e+07]
 [1.63961466e+06]
 [2.17580000e+07]
 [2.49189483e+06]
 [2.07882166e+07]
 [4.24342393e+06]
 [2.79248316e+08]
 [1.26820026e+07]
 [2.60130515e+08]
 [1.54865290e+07]
 [5.40897135e+05]
 [2.33351732e+06]
 [4.47692242e+07]
 [5.52208044e+05]
 [2.67451835e+06]
 [9.70060604e+05]
 [2.93158410e+06]
 [2.35140563e+07]
 [2.40507804e+06]
 [3.77540938e+06]
 [1.17290492e+08]
 [5.55563399e+06]
 [1.82188387e+08]
 [7.13051784e+05]
 [1.39404678e+08]
 [1.86144613e+08]
 [1.22117442e+07]
 [1.02947306e+08]
 [1.14305571e+06]
 [6.28794586e+05]
 [2.33883939e+07]
 [1.01229138e+07]
 [4.24581298e+06]
 [1.78643654e+05]
 [1.07825671e+07]
 [1.17841020e+07]
 [3.70509013e+07]
 [2.93393680e+07]
 [4.15007450e+07]
 [1.03442351e+08]
 [7.34282677e+07]
 [1.86865426e+06]
 [5.21916044e+07]
 [2.70690240e+07]
 [1.50270415e+07]
 [5.38061547e+07]
 [1.58270536e+08]
 [7.81259997e+07]
 [2.73505889e+07]
 [3.62990942e+08]
 [9.40846270e+07]
 [2.77380292e+07]
 [1.51598354e+07]
 [6.62198470e+07]
 [6.79132870e+07]
 [1.73922298e+08]
 [2.32784833e+07]
 [2.17363367e+07]
 [5.14767672e+07]
 [4.79271040e+07]
 [5.26083854e+07]
 [4.42015827e+07]
 [1.89067058e+07]
 [6.62185124e+07]
 [9.38476373e+07]
 [6.34167811e+07]
 [7.48900057e+07]
 [1.02742423e+08]
 [3.90576090e+07]
 [2.14690920e+07]
 [1.71169234e+08]
 [7.32279034e+06]
 [1.43516156e+07]
 [1.04382501e+08]
 [1.14842248e+08]]


#### SVM ####
0.6932270916334662
0.5625
[2 1 2 2 3 2 1 1 2 2 2 2 3 2 3 3]
[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

"""

if __name__ == "__main__":
    training_data, test_data = get_data()
    X = training_data["pigeon"] + training_data["pizza"]
    y = [0]*len(training_data["pigeon"]) + [1]*len(training_data["pizza"])

    print("#### KDA ####")
    s= dml.kda.KDA(n_components=2, kernel='poly', degree=2).fit(X,y)
    transform = s.transform(X)
    #plt.plot(transform/min(transform),'ro')
    min = min(transform)
    new_transform = transform/min
    plt.plot(new_transform[0:len(training_data["pigeon"])], 'ro')
    plt.plot(new_transform[len(training_data["pigeon"]):], 'bo')
    plt.show()
    print(transform)
    print("#### SVM ####")
    X = X + training_data["scorpion"] + training_data["sunflower"]
    y = y + [2]*len(training_data["scorpion"]) + [3]*len(training_data["sunflower"])
    clf = SVC()
    clf.fit(X, y)
    X_test = test_data["pigeon"] + test_data["pizza"] + test_data["scorpion"] + test_data["sunflower"]
    y_test = [0]*len(test_data["pigeon"]) + [1]*len(test_data["pizza"]) + [2]*len(test_data["scorpion"]) + [3]*len(test_data["sunflower"])
    print(clf.score(X, y))
    print(clf.score(X_test, y_test))
    print(clf.predict(X_test))
    print(y_test)

    print("#### KDA SVM ####")