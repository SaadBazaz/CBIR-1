# import packages
from __future__ import print_function

import numpy as np

from image_search_pipeline.descriptors import DetectAndDescribe
from image_search_pipeline.information_retrieval import BagOfVisualWords
from image_search_pipeline.information_retrieval import Searcher
from image_search_pipeline.information_retrieval import chi2_distance
from image_search_pipeline import ResultsMontage
from scipy.spatial import distance
from redis import Redis
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
import argparse
import pickle
import imutils
import json
import cv2

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory of indexed images")
ap.add_argument("-f", "--features_db", required = True, help = "Path to the feature database")
ap.add_argument("-b", "--bovw_db", required = True, help = "Path to the bag-of-visual-words database")
ap.add_argument("-c", "--codebook", required = True, help = "Path to the codebook")
ap.add_argument("-i", "--idf", type = str, help = "Path to inverted document frequencies array")
# ap.add_argument("-r", "--relevant", required = True, help = "Path to the relevant dictionary")
ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
args = vars(ap.parse_args())



dbs = args["features_db"].split(",")
codebooks = args["codebook"].split(",") 
bovw_dbs = args["bovw_db"].split(",") 
idfs = [None]*4
if args["idf"] is not None:
    idfs = args["idf"].split(",")

# load the relevant queries dictionary and lookup the relevant results for the
# query image

# relevant = json.loads(open(args["relevant"]).read())
queryFilename = args["query"][args["query"].rfind("/") + 1:]
# queryRelevant = relevant[queryFilename]

# load the query image and process it
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", imutils.resize(queryImage, width = 320))

print ("(original) queryImage shape is", queryImage.shape)

queryImage = imutils.resize(queryImage, width = 320)
queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

B = [M for SubRow in np.split(queryImage,2, axis = 0) for M in np.split(SubRow,2, axis = 1)]

C1,C2,C3,C4 = B

# Q1 = C1
# Q2 = np.concatenate( (C1, C2) )
# Q3 = np.concatenate( (C2, C4) )
# Q4 = image

quads = [
    C1, 
    C3, 
    C4, 
    C2
]

# connect to redis
redisDB = Redis(host = "localhost", port = 6379, db = 0)

final_time = []

for part_num, (queryImage, db, cb, bdb, _idf) in enumerate(zip(quads, dbs, codebooks, bovw_dbs, idfs)):

    # initialize the keypoint detector, local invariant descriptor, descriptor pipeline,
    # distance metric, and inverted document frequency array

    # detector = FeatureDetector_create("SURF")
    detector = cv2.ORB_create(nfeatures=1500)
    # descriptor = DescriptorExtractor_create("RootSIFT")
    descriptor = cv2.ORB_create(nfeatures=1500)

    dad = DetectAndDescribe(detector, descriptor)
    distanceMetric = chi2_distance
    idf = None

    # if the path to the inverted document frequency is array was supplied, then load
    # idf array and update the distance metric
    if _idf is not None:
        idf = pickle.loads(open(_idf, "rb").read())
        distanceMetric = distance.cosine

    # load the codebook vocabulary and initialize the BOVW transformer
    vocab = pickle.loads(open(cb, "rb").read())
    bovw = BagOfVisualWords(vocab)

    print ("queryImage shape is", queryImage.shape)

    cv2.imshow("Query " + str(part_num), imutils.resize(queryImage, width = 320))


    # queryImage = imutils.resize(queryImage, width = 320)
    # queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    # extract features from the query image and construct a bag-of-visual-word from  it
    (_, descs) = dad.describe(queryImage)
    hist = bovw.describe(descs).tocoo()

    print ("Hist is ", type(hist))

    # perform the search
    searcher = Searcher(redisDB, bdb, db, idf = idf,
        distanceMetric = distanceMetric)
    
    try:
        sr = searcher.search(hist, numResults = 20)
    except Exception as e:
        print("[ERROR]", str(e))
        continue

    print("[INFO] search took: {:.2f}s".format(sr.search_time))

    final_time.append(sr.search_time)

    print("Results are", sr.results)

    # initialize the results montage
    montage = ResultsMontage((240, 320), 5, 20)

    # loop over the individual results
    for (i, (score, resultID, resultsIdx)) in enumerate(sr.results):
        # load the result image and display it
        print("[RESULT] {result_num}. {result} - {score:.2f}".format(result_num = i + 1,
            result = resultID, score = score))
        result = cv2.imread("{}/{}".format(args["dataset"], resultID))
        montage.addResult(result, text = "#{}".format(i + 1),
            # highlight = resultID in queryRelevant
            )

    # show the output image of results
    cv2.imshow("Results", imutils.resize(montage.montage, height = 700))
    cv2.waitKey(0)

    searcher.finish()

print('=== FINAL STATS ===')
# print("Average Memory use:", final_avg_mem/4)
print("Average Time taken (per iteration):", sum(final_time)/len(final_time))
print("Time taken for early result set:", final_time[1])
print("Total Time taken:", sum(final_time))

cv2.destroyAllWindows()
