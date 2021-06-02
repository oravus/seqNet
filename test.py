import torch
from torch.utils.data import DataLoader
import faiss
from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import numpy as np
import time

def aggregateMatchScores(dbDesc,qDesc,device,refCandidates=None):
    numDb, numQ = dbDesc.shape[0], qDesc.shape[0]

    if refCandidates is None:
        shape = [numDb,numQ]
    else:
        shape = refCandidates.transpose().shape

    dMat_seq = -1*torch.ones(shape,device=device)

    for j in tqdm(range(numQ), total=numQ, leave=True):
        t1 = time.time()
        if refCandidates is not None:
            rCands = refCandidates[j]
        else:
            rCands = torch.arange(numDb)
        for i,r in enumerate(rCands):
            dMat = torch.cdist(dbDesc[r].unsqueeze(0),qDesc[j].unsqueeze(0))
            dMat_seq[i,j] = torch.diagonal(dMat,0,1,2).mean(-1)

    return dMat_seq.detach().cpu().numpy()

def getRecallAtN(n_values, predictions, gt):
    correct_at_n = np.zeros(len(n_values))
    numQWithoutGt = 0
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        if len(gt[qIx]) == 0:
            numQWithoutGt += 1
            continue
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    return correct_at_n / (len(gt)-numQWithoutGt)

def test(opt, model, encoder_dim, device, eval_set, writer, epoch=0, write_tboard=False, extract_noEval=False):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=not opt.nocuda)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'seqnet':
            pool_size = opt.outDims
        if 'seqmatch' in opt.pooling.lower():
            dbFeat = torch.empty((len(eval_set), opt.seqL, pool_size),device=device)
        else:
            dbFeat = torch.empty((len(eval_set), pool_size),device=device)

        durs_batch = []
        for iteration, (input, indices) in tqdm(enumerate(test_data_loader, 1),total=len(test_data_loader)-1, leave=False):
            t1 = time.time()
            input = input.float().to(device)
            if opt.pooling.lower() == 's1+seqmatch':
                shapeOrig = input.shape
                input = input.reshape([-1,input.shape[-1]])
                seq_encoding = model.pool(input).reshape(shapeOrig)
            else:
                seq_encoding = model.pool(input)
            if 'seqmatch' in opt.pooling.lower():
                dbFeat[indices,:,:] = seq_encoding
            else:
                dbFeat[indices, :] = seq_encoding
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)
            durs_batch.append(time.time()-t1)
            del input
    del test_data_loader
    print("Average batch time:", np.mean(durs_batch), np.std(durs_batch))

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:]
    dbFeat = dbFeat[:eval_set.dbStruct.numDb]
    print(dbFeat.shape, qFeat.shape)

    qFeat_np = qFeat.detach().cpu().numpy().astype('float32')
    dbFeat_np = dbFeat.detach().cpu().numpy().astype('float32')

    db_emb, q_emb = None, None
    if opt.numSamples2Project != -1 and not write_tboard:
        db_emb = TSNE(n_components=2).fit_transform(dbFeat_np[:opt.numSamples2Project])
        q_emb = TSNE(n_components=2).fit_transform(qFeat_np[:opt.numSamples2Project])

    if extract_noEval:
        return np.vstack([dbFeat_np,qFeat_np]), db_emb, q_emb, None, None

    n_values = [1,5,10,20,100]

    if 'seqmatch' in opt.pooling.lower():
        print('====> Performing sequence score aggregation')
        if opt.predictionsFile is not None:
            predPrior = np.load(opt.predictionsFile)['preds']
            predPriorTopK = predPrior[:,:20]
        else:
            predPriorTopK = None
        dMatSeq = aggregateMatchScores(dbFeat,qFeat,device,refCandidates=predPriorTopK)
        predictions = np.argsort(dMatSeq,axis=0)[:max(n_values),:].transpose()
        bestDists = dMatSeq[predictions[:,0],np.arange(dMatSeq.shape[1])]
        if opt.predictionsFile is not None:
            predictions = np.array([predPriorTopK[qIdx][predictions[qIdx]] for qIdx in range(predictions.shape[0])])
            print("Preds:",predictions.shape)
    else:
        print('====> Building faiss index')
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(dbFeat_np)

        distances, predictions = faiss_index.search(qFeat_np, max(n_values))
        bestDists = distances[:,0]

    print('====> Calculating recall @ N')

    # for each query get those within threshold distance
    gt,gtDists = eval_set.get_positives(retDists=True)
    gtDistsMat = cdist(eval_set.dbStruct.utmDb,eval_set.dbStruct.utmQ)

    # compute recall for different loc radii
    rAtL = []
    for locRad in [1,5,10,20,40,100,200]:
        gtAtL = gtDistsMat <= locRad
        gtAtL = [np.argwhere(gtAtL[:,qIx]).flatten() for qIx in range(gtDistsMat.shape[1])]
        rAtL.append(getRecallAtN(n_values, predictions, gtAtL))

    recall_at_n = getRecallAtN(n_values, predictions, gt)

    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard: writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    return recalls, db_emb, q_emb, rAtL, predictions