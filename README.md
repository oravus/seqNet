# SeqNet: Learning Descriptors for Sequence-Based Hierarchical Place Recognition

[[ArXiv+Supplementary](https://arxiv.org/abs/2102.11603)] [[IEEE Xplore RA-L 2021](https://ieeexplore.ieee.org/abstract/document/9382076/)] [[ICRA 2021 YouTube Video](https://www.youtube.com/watch?v=KYw7RhDfxY0)]

**and**

# SeqNetVLAD vs PointNetVLAD: Image Sequence vs 3D Point Clouds for Day-Night Place Recognition

[[ArXiv](https://arxiv.org/abs/2106.11481)] [[CVPR 2021 Workshop 3DVR](https://sites.google.com/view/cvpr2021-3d-vision-robotics/)]

<p align="center">
  <img src="./assets/seqnet.jpg">
    <br/><em>Sequence-Based Hierarchical Visual Place Recognition.</em>
</p>

## News:
**Aug 03** : Added Oxford dataset files and a [direct link](https://cloudstor.aarnet.edu.au/plus/s/8L7loyTZjK0FsWT) to download the Nordland dataset.

**Jun 23**: CVPR 2021 Workshop 3DVR paper, "SeqNetVLAD vs PointNetVLAD", now available on [arXiv](https://arxiv.org/abs/2106.11481).

**Jun 02**: SeqNet code release with the Nordland dataset.

## Setup (One time)
### Conda
```bash
conda create -n seqnet python=3.8 mamba -c conda-forge -y
conda activate seqnet
mamba install numpy pytorch=1.8.0 torchvision tqdm scikit-learn faiss tensorboardx h5py -c conda-forge -y
```

### Download
Run `bash download.sh` to download single image NetVLAD descriptors (3.4 GB) for the Nordland-clean dataset <sup>[[a]](#nordclean)</sup> and the Oxford dataset (0.3 GB), and Nordland-trained model files (1.5 GB) <sup>[[b]](#saveLoc)</sup>.

## Run

### Train
To train sequential descriptors through SeqNet on the Nordland dataset:
```python
python main.py --mode train --pooling seqnet --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --expName "w5"
```
or the Oxford dataset (set `--dataset oxford-pnv` for pointnetvlad-like data split as described in the [CVPR 2021 Workshop paper](https://arxiv.org/abs/2106.11481)):
```python
python main.py --mode train --pooling seqnet --dataset oxford-v1.0 --seqL 5 --w 3 --outDims 4096 --expName "w3"
```

To train transformed single descriptors through SeqNet:
```python
python main.py --mode train --pooling seqnet --dataset nordland-sw --seqL 1 --w 1 --outDims 4096 --expName "w1"
```

### Test
```python
python main.py --mode test --pooling seqnet --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun03_15-22-44_l10_w5/ 
```
The above will reproduce results for SeqNet (S5) as per [Supp. Table III on Page 10](https://arxiv.org/pdf/2102.11603.pdf).

<details>
  <summary>To obtain other results from the same table in the paper, expand this. </summary>
  
```python
# Raw Single (NetVLAD) Descriptor
python main.py --mode test --pooling single --dataset nordland-sf --seqL 1 --split test

# SeqNet (S1)
python main.py --mode test --pooling seqnet --dataset nordland-sf --seqL 1 --split test --resume ./data/runs/Jun03_15-07-46_l1_w1/

# Raw + Smoothing
python main.py --mode test --pooling smooth --dataset nordland-sf --seqL 5 --split test

# Raw + Delta
python main.py --mode test --pooling delta --dataset nordland-sf --seqL 5 --split test

# Raw + SeqMatch
python main.py --mode test --pooling single+seqmatch --dataset nordland-sf --seqL 5 --split test

# SeqNet (S1) + SeqMatch
python main.py --mode test --pooling s1+seqmatch --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun03_15-07-46_l1_w1/

# HVPR (S5 to S1)
# Run S5 first and save its predictions by specifying `resultsPath`
python main.py --mode test --pooling seqnet --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun03_15-22-44_l10_w5/ --resultsPath ./data/results/
# Now run S1 + SeqMatch using results from above (the timestamp of `predictionsFile` would be different in your case)
python main.py --mode test --pooling s1+seqmatch --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun03_15-07-46_l1_w1/ --predictionsFile ./data/results/Jun03_16-07-36_l5_0.npz

```
</details>

## Acknowledgement
The code in this repository is based on [Nanne/pytorch-NetVlad](https://github.com/Nanne/pytorch-NetVlad). Thanks to [Tobias Fischer](https://github.com/Tobias-Fischer) for his contributions to this code during the development of our project [QVPR/Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD).

## Citation
```
@article{garg2021seqnet,
  title={SeqNet: Learning Descriptors for Sequence-based Hierarchical Place Recognition},
  author={Garg, Sourav and Milford, Michael},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={3},
  pages={4305-4312},
  year={2021},
  publisher={IEEE},
  doi={10.1109/LRA.2021.3067633}
}

@misc{garg2021seqnetvlad,
  title={SeqNetVLAD vs PointNetVLAD: Image Sequence vs 3D Point Clouds for Day-Night Place Recognition},
  author={Garg, Sourav and Milford, Michael},
  howpublished={CVPR 2021 Workshop on 3D Vision and Robotics (3DVR)},
  month={Jun},
  year={2021},
}
```

#### Other Related Projects
[Patch-NetVLAD (2021)](https://github.com/QVPR/Patch-NetVLAD);
[Delta Descriptors (2020)](https://github.com/oravus/DeltaDescriptors);
[CoarseHash (2020)](https://github.com/oravus/CoarseHash);
[seq2single (2019)](https://github.com/oravus/seq2single);
[LoST (2018)](https://github.com/oravus/lostX)

<a name="nordclean">[a]<a> This is the clean version of the dataset that excludes images from the tunnels and red lights and can be downloaded from [here](https://cloudstor.aarnet.edu.au/plus/s/8L7loyTZjK0FsWT).

<a name="saveLoc">[b]<a> These will automatically save to `./data/`, you can modify this path in [download.sh](https://github.com/oravus/seqNet/blob/main/download.sh) and [get_datasets.py](https://github.com/oravus/seqNet/blob/5450829c4294fe1d14966bfa1ac9b7c93237369b/get_datasets.py#L6) to specify your workdir.
