import cv2
import numpy as np
import glob
import sklearn
import sklearn.cluster
import matplotlib.pyplot as plt
#128*nbpts

def vocabulaire(N, chemins, fichier=None, methode = "kmeans"):
    s=cv2.SIFT_create()
    listdesc = []
    for chemin in chemins:
        for image in glob.glob(chemin):
            #print(image)
            img=cv2.imread(image)
            kpt, descriptor = s.detectAndCompute(img, None)
            listdesc.append(descriptor) 

    if methode == "kmeans":
        array = np.concatenate(listdesc)
        kmeans = sklearn.cluster.KMeans(N).fit(array)
        max = 0

        for desc in listdesc:
            try:
                label = kmeans.fit_predict(desc)
            except:
                print("erreur img " + str(listdesc.index(desc)))
                continue
            dist = np.linalg.norm(kmeans.cluster_centers_[label]-desc)
            if dist > max:
                max = dist
    if fichier is not None:
        np.savetxt(fichier, kmeans.cluster_centers_, delimiter=',')

    return kmeans.inertia_/N, max

def coude():
    N= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    inertie = []
    max = []
    for n in N:
        chemins = ["Img/pigeon/test/*.jpg", "Img/pizza/test/*.jpg", "Img/scorpion/test/*.jpg", "Img/sunflower/test/*.jpg"]
        inertia, max_dist = vocabulaire(n, chemins)
        inertie.append(inertia)
        max.append(max_dist)
    plt.figure(1)
    plt.plot(N, inertie)
    plt.figure(2)
    plt.plot(N, max)
    plt.show()
    input("Press Enter to continue...")

coude()