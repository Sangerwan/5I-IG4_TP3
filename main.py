import cv2
import numpy as np
import glob
import sklearn



def vocabulaire(N, chemins, fichier, methode = "kmeans"):
    s=cv2.SIFT_create()
    listkpts = []
    listdesc = []
    for chemin in chemins:
        for image in glob.glob(chemin):
            print(image)
            img=cv2.imread(image)
            kpt, descriptor = s.detectAndCompute(img, None)
            listkpts.append(kpt)
            listdesc.append(descriptor) 

    if methode == "kmeans":
        kmeans = sklearn.cluster.KMeans(N).fit(np.array(listdesc))
        kmeans.cluster_centers_
        kmeans.labels_
        max = 0
        for i in range (len(kmeans.labels_)):
            tmp = (listkpts(i).pt - kmeans.cluster_centers(kmeans.labels_(i))) * (listkpts(i).pt - kmeans.cluster_centers(kmeans.labels_(i)))
            if tmp > max:
                max = tmp
    if fichier is not None:
        #numpy.savetxt(fichier, )
        with open(fichier, "w") as f:
            for kpt in listkpts:
                f.write(kpt.pt)

    return kmeans.inertia_/N, max


N = 100
chemins = ["Img/pigeon/*.jpg", "Img/pizza/*.jpg", "Img/scorpion/*.jpg", "Img/sunflower/*.jpg"]
fichier = "vocabulaire.txt"
vocabulaire(N, chemins, fichier)
print("aaaa")