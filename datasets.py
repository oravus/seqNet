import torch
import torch.utils.data as data
import itertools

import os
from os.path import join, exists
from scipy.io import loadmat, savemat
import numpy as np
from collections import namedtuple

from sklearn.neighbors import NearestNeighbors
import faiss
import h5py

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr',
                                   'dbTimeStamp', 'qTimeStamp', 'gpsDb', 'gpsQ'])

class Dataset():
    def __init__(self, dataset_name, train_mat_file, test_mat_file, val_mat_file, opt):
        self.dataset_name = dataset_name
        self.train_mat_file = train_mat_file
        self.test_mat_file = test_mat_file
        self.val_mat_file = val_mat_file
        self.struct_dir = "./structFiles/"
        self.seqL = opt.seqL
        self.seqL_filterData = opt.seqL_filterData

        # descriptor settings
        self.dbDescs = None
        self.qDescs = None
        self.trainInds = None
        self.valInds = None
        self.testInds = None
        self.db_seqBounds = None
        self.q_seqBounds = None

    def loadPreComputedDescriptors(self,ft1,ft2,seqBounds=None):
        self.dbDescs = ft1
        self.qDescs = ft2
        print("All Db descs: ", self.dbDescs.shape)
        print("All Qry descs: ", self.qDescs.shape)
        if seqBounds is None:
            self.db_seqBounds = None
            self.q_seqBounds = None
        else:
            self.db_seqBounds = seqBounds[0]
            self.q_seqBounds = seqBounds[1]
        return self.dbDescs.shape[1]

    def get_whole_training_set(self, onlyDB=False):
        structFile = join(self.struct_dir, self.train_mat_file)
        indsSplit = self.trainInds
        return WholeDatasetFromStruct( structFile, indsSplit, self.dbDescs, self.qDescs, seqL=self.seqL, onlyDB=onlyDB, seqBounds=[self.db_seqBounds,self.q_seqBounds],seqL_filterData=self.seqL_filterData)

    def get_whole_val_set(self):
        structFile = join(self.struct_dir, self.val_mat_file)
        indsSplit = self.valInds
        if self.seqL_filterData is None and self.dataset_name == 'msls':
            self.seqL_filterData = self.seqL
        return WholeDatasetFromStruct(structFile, indsSplit, self.dbDescs, self.qDescs, seqL=self.seqL, seqBounds=[self.db_seqBounds,self.q_seqBounds],seqL_filterData=self.seqL_filterData)

    def get_whole_test_set(self):
        if self.test_mat_file is not None:
            structFile = join(self.struct_dir, self.test_mat_file)
            indsSplit = self.testInds
            if self.seqL_filterData is None and self.dataset_name == 'msls':
                self.seqL_filterData = self.seqL
            return WholeDatasetFromStruct(structFile, indsSplit, self.dbDescs, self.qDescs, seqL=self.seqL, seqBounds=[self.db_seqBounds,self.q_seqBounds],seqL_filterData=self.seqL_filterData)
        else:
            raise ValueError('test set not available for dataset ' + self.dataset_name)

    def get_training_query_set(self, margin=0.1, nNegSample=1000, use_regions=False):
        structFile = join(self.struct_dir, self.train_mat_file)
        indsSplit = self.trainInds
        return QueryDatasetFromStruct(structFile,indsSplit, self.dbDescs, self.qDescs, nNegSample=nNegSample, margin=margin,use_regions=use_regions, seqL=self.seqL, seqBounds=[self.db_seqBounds,self.q_seqBounds])

    def get_val_query_set(self):
        structFile = join(self.struct_dir, self.val_mat_file)
        indsSplit = self.valInds
        return QueryDatasetFromStruct(structFile, indsSplit, self.dbDescs, self.qDescs, seqL=self.seqL, seqBounds=[self.db_seqBounds,self.q_seqBounds])

    @staticmethod
    def collate_fn(batch):
        """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

        Args:
            batch: list of tuple (query, positive, negatives).
                - query: torch tensor of shape (T, C). e.g. (5,4096)
                - positive: torch tensor of shape (T, C).
                - negative: torch tensor of shape (N, T, C).
        Returns:
            query: torch tensor of shape (batch_size, T, C).
            positive: torch tensor of shape (batch_size, T, C).
            negatives: torch tensor of shape (batch_size, T, C).
        """

        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None

        query, positive, negatives, indices = zip(*batch)

        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
        negatives = torch.cat(negatives, 0)
        indices = list(itertools.chain(*indices))

        return query, positive, negatives, negCounts, indices

def getSeqInds(idx,seqL,maxNum,minNum=0,retLenDiff=False):
    seqLOrig = seqL
    seqInds = np.arange(max(minNum,idx-seqL//2),min(idx+seqL-seqL//2,maxNum),1)
    lenDiff = seqLOrig - len(seqInds)
    if retLenDiff:
        return lenDiff

    if seqInds[0] == minNum:
        seqInds = np.concatenate([seqInds,np.arange(seqInds[-1]+1,seqInds[-1]+1+lenDiff,1)])
    elif lenDiff > 0 and seqInds[-1] in range(maxNum-1,maxNum):
        seqInds = np.concatenate([np.arange(seqInds[0]-lenDiff,seqInds[0],1),seqInds])
    return seqInds

def getValidSeqInds(seqBounds,seqL):
    validFlags = []
    for i in range(len(seqBounds)):
        sIdMin, sIdMax = seqBounds[i]
        lenDiff = getSeqInds(i,seqL,sIdMax,minNum=sIdMin,retLenDiff=True)
        validFlags.append(True if lenDiff == 0 else False)
    return validFlags

def parse_db_struct(path):
    mat = loadmat(path)

    fieldnames = list(mat['dbStruct'][0, 0].dtype.names)

    dataset = mat['dbStruct'][0, 0]['dataset'].item()
    whichSet = mat['dbStruct'][0, 0]['whichSet'].item()

    dbImage = [f[0].item() for f in mat['dbStruct'][0, 0]['dbImageFns']]
    qImage = [f[0].item() for f in mat['dbStruct'][0, 0]['qImageFns']]

    numDb = mat['dbStruct'][0, 0]['numImages'].item()
    numQ = mat['dbStruct'][0, 0]['numQueries'].item()

    posDistThr = mat['dbStruct'][0, 0]['posDistThr'].item()
    posDistSqThr = mat['dbStruct'][0, 0]['posDistSqThr'].item()
    if 'nonTrivPosDistSqThr' in fieldnames:
        nonTrivPosDistSqThr = mat['dbStruct'][0, 0]['nonTrivPosDistSqThr'].item()
    else:
        nonTrivPosDistSqThr = None

    if 'dbTimeStamp' in fieldnames and 'qTimeStamp' in fieldnames:
        dbTimeStamp = [f[0].item() for f in mat['dbStruct'][0, 0]['dbTimeStamp'].T]
        qTimeStamp = [f[0].item() for f in mat['dbStruct'][0, 0]['qTimeStamp'].T]
        dbTimeStamp = np.array(dbTimeStamp)
        qTimeStamp = np.array(qTimeStamp)
    else:
        dbTimeStamp = None
        qTimeStamp = None

    if 'utmQ' in fieldnames and 'utmDb' in fieldnames:
        utmDb = mat['dbStruct'][0, 0]['utmDb'].T
        utmQ = mat['dbStruct'][0, 0]['utmQ'].T
    else:
        utmQ = None
        utmDb = None

    if 'gpsQ' in fieldnames and 'gpsDb' in fieldnames:
        gpsDb = mat['dbStruct'][0, 0]['gpsDb'].T
        gpsQ = mat['dbStruct'][0, 0]['gpsQ'].T
    else:
        gpsQ = None
        gpsDb = None

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr,
                    posDistSqThr, nonTrivPosDistSqThr, dbTimeStamp, qTimeStamp, gpsQ, gpsDb)


def save_db_struct(path, db_struct):
    assert db_struct.numDb == len(db_struct.dbImage)
    assert db_struct.numQ == len(db_struct.qImage)

    inner_dict = {
        'whichSet': db_struct.whichSet,
        'dbImageFns': np.array(db_struct.dbImage, dtype=np.object).reshape(-1, 1),
        'qImageFns': np.array(db_struct.qImage, dtype=np.object).reshape(-1, 1),
        'numImages': db_struct.numDb,
        'numQueries': db_struct.numQ,
        'posDistThr': db_struct.posDistThr,
        'posDistSqThr': db_struct.posDistSqThr,
    }

    if db_struct.dataset is not None:
        inner_dict['dataset'] = db_struct.dataset

    if db_struct.nonTrivPosDistSqThr is not None:
        inner_dict['nonTrivPosDistSqThr'] = db_struct.nonTrivPosDistSqThr

    if db_struct.utmDb is not None and db_struct.utmQ is not None:
        assert db_struct.numDb == len(db_struct.utmDb)
        assert db_struct.numQ == len(db_struct.utmQ)
        inner_dict['utmDb'] = db_struct.utmDb.T
        inner_dict['utmQ'] = db_struct.utmQ.T

    if db_struct.gpsDb is not None and db_struct.gpsQ is not None:
        assert db_struct.numDb == len(db_struct.gpsDb)
        assert db_struct.numQ == len(db_struct.gpsQ)
        inner_dict['gpsDb'] = db_struct.gpsDb.T
        inner_dict['gpsQ'] = db_struct.gpsQ.T

    if db_struct.dbTimeStamp is not None and db_struct.qTimeStamp is not None:
        inner_dict['dbTimeStamp'] = db_struct.dbTimeStamp.astype(np.float64)
        inner_dict['qTimeStamp'] = db_struct.qTimeStamp.astype(np.float64)

    savemat(path, {'dbStruct': inner_dict})

def print_db_concise(db):
    [print('\033[1m' + k + '\033[0m', v[:10] if type(v) is list else v) for k,v in db._asdict().items()]

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, indsSplit, dbDescs, qDescs, onlyDB=False, seqL=1, seqBounds=None,seqL_filterData=None):
        super().__init__()

        self.seqL = seqL
        self.filterBoundaryInds = False if seqL_filterData is None else True

        self.dbStruct = parse_db_struct(structFile)

        self.images = dbDescs[indsSplit[0]]

        if seqBounds[0] is None:
            self.seqBounds = np.array([[0,len(self.images)] for _ in range(len(self.images))])

        if not onlyDB:
            qImages = qDescs[indsSplit[1]]
            self.images = np.concatenate([self.images,qImages],0)
            if seqBounds[0] is None:
                q_seqBounds = np.array([[len(self.seqBounds),len(self.images)] for _ in range(len(qImages))])
                self.seqBounds = np.vstack([self.seqBounds,q_seqBounds])

        if seqBounds[0] is not None:
            db_seqBounds = seqBounds[0][indsSplit[0]]
            q_seqBounds = db_seqBounds[-1,-1] + seqBounds[1][indsSplit[1]]
            self.seqBounds = np.vstack([db_seqBounds,q_seqBounds])

        self.validInds = np.arange(len(self.images))
        self.validInds_db = np.arange(self.dbStruct.numDb)
        self.validInds_q = np.arange(self.dbStruct.numQ)
        if self.filterBoundaryInds:
            validFlags = getValidSeqInds(self.seqBounds,seqL_filterData)
            self.validInds = np.argwhere(validFlags).flatten()
            self.validInds_db = np.argwhere(validFlags[:self.dbStruct.numDb]).flatten()
            self.validInds_q = np.argwhere(validFlags[self.dbStruct.numDb:]).flatten()
            self.dbStruct = self.dbStruct._replace(utmDb=self.dbStruct.utmDb[self.validInds_db], numDb=len(self.validInds_db), utmQ=self.dbStruct.utmQ[self.validInds_q], numQ=len(self.validInds_q))

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        origIndex = index
        index = self.validInds[index]
        sIdMin, sIdMax = self.seqBounds[index]
        img = self.images[getSeqInds(index,self.seqL,sIdMax,minNum=sIdMin)]

        return img, origIndex

    def __len__(self):
        return len(self.validInds)

    def get_positives(self,retDists=False):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            print("Using Localization Radius: ", self.dbStruct.posDistThr)
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.posDistThr)

        if retDists:
            return self.positives, self.distances
        else:
            return self.positives


class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, indsSplit, dbDescs, qDescs, nNegSample=1000, nNeg=10, margin=0.1, use_regions=False, seqL=1, seqBounds=None):
        super().__init__()

        self.seqL = seqL

        self.dbDescs = dbDescs[indsSplit[0]]
        self.qDescs = qDescs[indsSplit[1]]

        self.margin = margin

        self.dbStruct = parse_db_struct(structFile)

        if seqBounds[0] is None:
            self.db_seqBounds = np.array([[0,len(self.dbDescs)] for _ in range(len(self.dbDescs))])
            self.q_seqBounds = np.array([[0,len(self.qDescs)] for _ in range(len(self.qDescs))])
        else:
            self.db_seqBounds = seqBounds[0][indsSplit[0]]
            self.q_seqBounds = seqBounds[1][indsSplit[1]]
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample  # number of negatives to randomly sample
        self.nNeg = nNeg  # number of negatives used for training
        self.use_faiss = True
        self.use_regions = use_regions

        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        self.nontrivial_distances, self.nontrivial_positives = \
            knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5,
                                 return_distance=True)

        self.nontrivial_positives = list(self.nontrivial_positives)

        # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)

        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]
        print("\n Queries within range ",len(self.queries), len(self.nontrivial_positives),"\n")

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                   radius=self.dbStruct.posDistThr,
                                                   return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True))

        self.cache = None  # filepath of HDF5 containing feature vectors for images
        self.h5feat = None

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index + qOffset]

            posFeat = h5feat[self.nontrivial_positives[index].tolist()]

            if self.use_faiss:
                faiss_index = faiss.IndexFlatL2(posFeat.shape[1])
                # noinspection PyArgumentList
                faiss_index.add(posFeat)
                # noinspection PyArgumentList
                dPos, posNN = faiss_index.search(qFeat.reshape(1, -1), 1)#posFeat.shape[0])
                dPos = np.sqrt(dPos)  # faiss returns squared distance
            else:
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(posFeat)
                dPos, posNN = knn.kneighbors(qFeat.reshape(1, -1), 1)#posFeat.shape[0])
            if len(self.nontrivial_positives[index]) < 1:
                # if none are violating then skip this query
                return None
            dPos = dPos[0][-1].item()
            posIndex = self.nontrivial_positives[index][posNN[0,-1]].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))
            negSample = np.sort(negSample) #essential to order ascending, speeds up h5 by about double

            negFeat = h5feat[negSample.astype(int).tolist()]
            if self.use_faiss:
                faiss_index = faiss.IndexFlatL2(posFeat.shape[1])
                # noinspection PyArgumentList
                faiss_index.add(negFeat)
                # noinspection PyArgumentList
                dNeg, negNN = faiss_index.search(qFeat.reshape(1, -1), self.nNeg * 10)
                dNeg = np.sqrt(dNeg)
            else:
                knn.fit(negFeat)

                # to quote netvlad paper code: 10x is hacky but fine
                dNeg, negNN = knn.kneighbors(qFeat.reshape(1, -1), self.nNeg * 10)

            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg < dPos + self.margin**0.5

            if np.sum(violatingNeg) < 1:
                # if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        sIdMin_q, sIdMax_q = self.q_seqBounds[index]
        query = self.qDescs[getSeqInds(index,self.seqL,sIdMax_q,sIdMin_q)]
        sIdMin_p, sIdMax_p = self.db_seqBounds[posIndex]
        positive = self.dbDescs[getSeqInds(posIndex,self.seqL,sIdMax_p,sIdMin_p)]

        negatives = []
        for negIndex in negIndices:
            sIdMin_n, sIdMax_n = self.db_seqBounds[negIndex]
            negative = torch.tensor(self.dbDescs[getSeqInds(negIndex,self.seqL,sIdMax_n,sIdMin_n)])
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        # noinspection PyTypeChecker
        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

    def __len__(self):
        return len(self.qDescs)
