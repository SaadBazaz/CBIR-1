# import packages
from image_search_pipeline.information_retrieval import BagOfVisualWords
from image_search_pipeline.indexer import BOVWIndexer
import argparse
import pickle
import h5py

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features_db", required = True,
    help = "Path to the features database")
ap.add_argument("-c", "--codebook", required = True,
    help = "Path to the codebook")
ap.add_argument("-b", "--bovw_db", required = True,
    help = "Path to where the bag-of-visual-words database will be stored")
ap.add_argument("-d", "--idf", required = True,
    help = "Path to inverse document frequency counts will be stored")
ap.add_argument("-s", "--max_buffer_size", type = int, default = 500,
    help = "Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

dbs = args["features_db"].split(",")
codebooks = args["codebook"].split(",") 
bovw_dbs = args["bovw_db"].split(",") 
idfs = args["idf"].split(",")

for part_num, (db, cb, bdb, idf) in enumerate(zip(dbs, codebooks, bovw_dbs, idfs)):

    print ("Part number {}...".format(part_num))

    # load the codebook vocabulary and initialize the bag-of-visual-words transformer
    vocab = pickle.loads(open(cb, "rb").read())
    bovw = BagOfVisualWords(vocab)

    # open the features database and initialize the bag-of-visual-words indexer
    featuresDB = h5py.File(db, mode = "r")
    bi = BOVWIndexer(bovw.codebook.shape[0], bdb,
        estNumImages = featuresDB["image_ids"].shape[0],
        maxBufferSize = args["max_buffer_size"])

    # loop over the image IDs and index
    for (i, (imageID, offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):
        # check to see if progress should be displayed
        if i > 0 and i % 10 == 0:
            bi._debug("processed {} images".format(i), msgType = "[PROGRESS]")

        # extract the feature vectors for the current image using the starting and
        # ending offsets (while ignoring the keypoints) and then quantize the
        # features to construct the bag-of-visual-words histogram
        features = featuresDB["features"][offset[0] : offset[1]][:, 2:]
        hist = bovw.describe(features)

        # add the bag-of-visual-words to the index
        bi.add(hist)

    # close the features database and finish the indexing process
    featuresDB.close()
    bi.finish()

    # dump the inverse dcument fequency counts to file
    f = open(idf, "wb")
    f.write(pickle.dumps(bi.df(method = "idf")))
    f.close()
