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
**Jan 27, 2024** : Download all pretrained models from [here](https://universityofadelaide.box.com/s/mp45yapl0j0by6aijf5kj8obt8ky0swk), Nordland dataset from [here](https://universityofadelaide.box.com/s/zkfk1akpbo5318fzqmtvlpp7030ex4up) and precomputed descriptors from [here](https://universityofadelaide.box.com/s/p8uh5yncsaxk7g8lwr8pihnwkqbc2pkf)

**Jan 18, 2022** : MSLS training setup included.

**Jan 07, 2022** : Single Image Vanilla NetVLAD feature extraction enabled.

**Oct 13, 2021** : ~~Oxford & Brisbane Day-Night pretrained models [download link](https://cloudstor.aarnet.edu.au/plus/s/wx0zIGi3WBTtq5F).~~ (use the latest link provided above)

**Aug 03, 2021** : Added Oxford dataset files ~~and a [direct link](https://cloudstor.aarnet.edu.au/plus/s/8L7loyTZjK0FsWT) to download the Nordland dataset.~~ (use the latest link provided above)

**Jun 23, 2021**: CVPR 2021 Workshop 3DVR paper, "SeqNetVLAD vs PointNetVLAD", now available on [arXiv](https://arxiv.org/abs/2106.11481).

## Setup
### Conda
```bash
conda create -n seqnet numpy pytorch=1.8.0 torchvision tqdm scikit-learn faiss tensorboardx h5py -c pytorch -c conda-forge
```

### Download
~~Run `bash download.sh` to download single image NetVLAD descriptors (3.4 GB) for the Nordland-clean dataset <sup>[[a]](#nordclean)</sup> and the Oxford dataset (0.3 GB), and Nordland-trained model files (1.5 GB) <sup>[[b]](#saveLoc)</sup>. Other pre-trained models for Oxford and Brisbane Day-Night can be downloaded from [here](https://universityofadelaide.box.com/s/mp45yapl0j0by6aijf5kj8obt8ky0swk).~~ [Please see download links at the top news from 27 Jan 2024]

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
or the MSLS dataset (specifying `--msls_trainCity` and `--msls_valCity` as default values):
```python
python main.py --mode train --pooling seqnet --dataset msls --msls_trainCity melbourne --msls_valCity austin --seqL 5 --w 3 --outDims 4096 --expName "msls_w3"
```

To train transformed single descriptors through SeqNet:
```python
python main.py --mode train --pooling seqnet --dataset nordland-sw --seqL 1 --w 1 --outDims 4096 --expName "w1"
```

### Test
On the Nordland dataset:
```python
python main.py --mode test --pooling seqnet --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun03_15-22-44_l10_w5/ 
```
On the MSLS dataset (can change `--msls_valCity` to `melbourne` or `austin` too):
```python
python main.py --mode test --pooling seqnet --dataset msls --msls_valCity amman --seqL 5 --split test --resume ./data/runs/<modelName>/
```

The above will reproduce results for SeqNet (S5) as per [Supp. Table III on Page 10](https://arxiv.org/pdf/2102.11603.pdf).

<details>
  <summary> [Expand this] To obtain other results from the same table in the paper, expand this. </summary>
  
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

### Single Image Vanilla NetVLAD Extraction
<details>
<summary> [Expand this] To obtain the single image vanilla NetVLAD descriptors (i.e. the provided precomputed .npy descriptors) </summary>

```bash
# Setup Patch-NetVLAD submodule from the seqNet repo:
cd seqNet 
git submodule update --init

# Download NetVLAD+PCA model
cd thirdparty/Patch-NetVLAD/patchnetvlad/pretrained_models
wget -O pitts_orig_WPCA4096.pth.tar https://huggingface.co/TobiasRobotics/Patch-NetVLAD/resolve/main/pitts_WPCA4096.pth.tar?download=true

# Compute global descriptors
cd ../../../Patch-NetVLAD/
python feature_extract.py --config_path patchnetvlad/configs/seqnet.ini --dataset_file_path ../../structFiles/imageNamesFiles/oxford_2014-12-16-18-44-24_imagenames_subsampled-2m.txt --dataset_root_dir <PATH_TO_OXFORD_IMAGE_DIR> --output_features_fullpath ../../data/descData/netvlad-pytorch/oxford_2014-12-16-18-44-24_stereo_left.npy

# example for MSLS (replace 'database' with 'query' and use different city names to compute all)
python feature_extract.py --config_path patchnetvlad/configs/seqnet.ini --dataset_file_path ../../structFiles/imageNamesFiles/msls_melbourne_database_imageNames.txt --dataset_root_dir <PATH_TO_Mapillary_Street_Level_Sequences> --output_features_fullpath ../../data/descData/netvlad-pytorch/msls_melbourne_database.npy
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
[SeqMatchNet (2021)](https://github.com/oravus/SeqMatchNet);
[Patch-NetVLAD (2021)](https://github.com/QVPR/Patch-NetVLAD);
[Delta Descriptors (2020)](https://github.com/oravus/DeltaDescriptors);
[CoarseHash (2020)](https://github.com/oravus/CoarseHash);
[seq2single (2019)](https://github.com/oravus/seq2single);
[LoST (2018)](https://github.com/oravus/lostX)

<a name="nordclean">[a]<a> This is the clean version of the dataset that excludes images from the tunnels and red lights and can be downloaded from [here](https://universityofadelaide.app.box.com/s/zkfk1akpbo5318fzqmtvlpp7030ex4up).

<a name="saveLoc">[b]<a> These will automatically save to `./data/`, you can modify this path in [download.sh](https://github.com/oravus/seqNet/blob/main/download.sh) and [get_datasets.py](https://github.com/oravus/seqNet/blob/5450829c4294fe1d14966bfa1ac9b7c93237369b/get_datasets.py#L6) to specify your workdir.
