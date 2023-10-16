from pyclustering.cluster.bsas import bsas
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.cure import cure
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.ema import ema
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.mbsas import mbsas
from pyclustering.cluster.optics import optics
from pyclustering.cluster.rock import rock
from pyclustering.cluster.somsc import somsc
from pyclustering.cluster.syncsom import syncsom
from pyclustering.cluster.ttsas import ttsas
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, MiniBatchKMeans, MeanShift, AffinityPropagation
from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.bang import bang

def ManualUp(X):
    fname = 'ManualUp'
    pred = [0]*len(X)
    return pred, fname

def KmeansCluster(n, X):
    fname = 'Kmeans'
    pred = KMeans(n).fit_predict(X)
    return pred, fname

def AgglomerativeCluster(n, X):
    fname = 'Agglomerative'
    pred = AgglomerativeClustering(n).fit_predict(X)
    return pred, fname

def BirchCluster(n, X):
    fname = 'Birch'
    pred = Birch(n).fit_predict(X)
    return pred, fname

def KmedoidsCluster(n, X):
    fname = 'Kmedoids'
    kmedoids = KMedoids(n).fit(X)
    pred = kmedoids.predict(X)
    return pred, fname

def MiniBatchKMeansCluster(n, X):
    fname = 'MiniBatchKmeans'
    pred = MiniBatchKMeans(n).fit_predict(X)
    return pred, fname

def MeanShiftCluster(X):
    fname = 'MeanShift'
    pred = MeanShift().fit_predict(X)
    return pred, fname

def AffinityPropagationCluster(X):
    fname = 'AP'
    pred = AffinityPropagation().fit_predict(X)
    return pred, fname


def BsasCluster(n, parameter, X):
    # Create instance of BSAS algorithm.
    fname = 'Bsas'
    bsas_instance = bsas(X, n, parameter)
    bsas_instance.process()
    # Get clustering results.
    clusters = bsas_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def CureCluster(n, X):
    fname = 'Cure'
    cure_instance = cure(X, n)
    cure_instance.process()
    clusters = cure_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def DbscanCluster(n, threshod, X):
    fname = 'Dbscan'
    dbscan_instance = dbscan(X, threshod, n)
    dbscan_instance.process()
    clusters = dbscan_instance.get_clusters()
    #noise = dbscan_instance.get_noise()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def MbsasCluster(n, threshold, X):
    fname = 'Mbsas'
    mbsas_instance = mbsas(X, n, threshold)
    mbsas_instance.process()
    clusters = mbsas_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def OpticsCluster(n, threshold, X):
    fname = 'Optics'
    optics_instance = optics(X, threshold, n)
    optics_instance.process()
    clusters = optics_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def RockCluster(n, thredshold, X):
    fname = 'Rock'
    rock_instance = rock(X, 1.0, 7)
    rock_instance.process()
    clusters = rock_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def Somsc(n, X):
    fname = 'Somsc'
    somsc_instance = somsc(X, n)
    somsc_instance.process()
    pred = somsc_instance.predict(X)
    return pred, fname

def SyncsomCluster(X):
    fname = 'Syncsom'
    network = syncsom(X, 4, 4, 1.0)
    network.process()
    clusters = network.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def BangCluster(X):
    fname = 'Bang'
    level = 3
    bangc = bang(X, level)
    bangc.process()
    clusters = bangc.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def Kmeans_plusplusCluster(X):
    fname = 'KmeansPlus'
    centers = kmeans_plusplus_initializer(X, 2, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
    # Perform cluster analysis using K-Means algorithm with initial centers.
    kmeans_instance = kmeans(X, centers)
    # Run clustering process and obtain result.
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def ClaransCluster(X):
    fname = 'clarans'
    clarans_instance = clarans(X, 2, 10, 5)
    clarans_instance.process()
    clusters = clarans_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname


def EmaCluster(X):
    fname = 'EMA'
    # Create EM algorithm to allocated four clusters.
    ema_instance = ema(X, 3)
    # Run clustering process.
    ema_instance.process()
    # Get clustering results.
    clusters = ema_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def FcmCluster(X):
    fname = 'Fcm'
    initial_centers = kmeans_plusplus_initializer(X, 2,
                                                  kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
    # create instance of Fuzzy C-Means algorithm
    fcm_instance = fcm(X, initial_centers)
    # run cluster analysis and obtain results
    fcm_instance.process()
    clusters = fcm_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def GmeansCluster(X):
    fname = 'Gmeans'
    # Create instance of G-Means algorithm. By default algorithm start search from single cluster.
    gmeans_instance = gmeans(X, repeat=10).process()
    # Extract clustering results: clusters and their centers
    clusters = gmeans_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname


def TtsasCluster(X):
    fname = 'Ttsas'
    # Prepare algorithm's parameters.
    threshold1 = 1.0
    threshold2 = 2.0
    # Create instance of TTSAS algorithm.
    ttsas_instance = ttsas(X, threshold1, threshold2)
    ttsas_instance.process()
    # Get clustering results.
    clusters = ttsas_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

def Xmeans(X):
    fname = 'Xmeans'
    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(X, amount_initial_centers).initialize()
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is 20.
    xmeans_instance = xmeans(X, initial_centers, 20)
    xmeans_instance.process()
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    pred = [0] * len(X)
    if clusters:
        for i in range(len(clusters)):
            for j in clusters[i]:
                pred[j] = i
    return pred, fname

