import cv2
import numpy as np
import glob
import sklearn
import sklearn.cluster
#128*nbpts

def vocabulaire(N, chemins, fichier, methode = "kmeans"):
    s=cv2.SIFT_create()
    listdesc = []
    for chemin in chemins:
        for image in glob.glob(chemin):
            print(image)
            img=cv2.imread(image)
            kpt, descriptor = s.detectAndCompute(img, None)
            listdesc.append(descriptor) 

    if methode == "kmeans":
        array = np.concatenate(listdesc)
        kmeans = sklearn.cluster.KMeans(N).fit(array)
        kmeans.cluster_centers_
        kmeans.labels_
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
        #numpy.savetxt(fichier, )
        with open(fichier, "w") as f:
            f.write(kmeans.cluster_centers_)

    return kmeans.inertia_/N, max


N = 100
chemins = ["Img/pigeon/*.jpg", "Img/pizza/*.jpg", "Img/scorpion/*.jpg", "Img/sunflower/*.jpg"]
fichier = "vocabulaire.txt"
vocabulaire(N, chemins, fichier)
print("aaaa")