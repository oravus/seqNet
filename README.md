# SeqNet: Learning Descriptors for Sequence-Based Hierarchical Place Recognition

[[arXiv pre-print with supplementary material](https://arxiv.org/abs/2102.11603)] [[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9382076/)]

<p align="center">
  <img src="./assets/seqnet.jpg">
    <br/><em>Sequence-Based Hierarchical Visual Place Recognition.</em>
</p>

# Run

## Download
Run `bash download.sh` to download single image NetVLAD descriptors (3.4 GB) for the Nordland-clean dataset <sup>[[a]](#nordclean)</sup> and corresponding model files (1.5 GB). These will automatically save to `./data/`, you can modify this path in [download.sh](https://github.com/oravus/seqNet/blob/main/download.sh) and [get_datasets.py](https://github.com/oravus/seqNet/blob/5450829c4294fe1d14966bfa1ac9b7c93237369b/get_datasets.py#L6) to specify your workdir.

## Train
To train sequential descriptors through SeqNet:
```python
python main.py --mode train --pooling seqnet --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --expName "l10_w5"
```

To (re-)train single descriptors through SeqNet:
```python
python main.py --mode train --pooling seqnet --dataset nordland-sw --seqL 1 --w 1 --outDims 4096 --expName "l1_w1"
```

## Test
```python
python main.py --mode test --pooling seqnet --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun02_00-08-23_l10_l10w5/ 
```
The above will reproduce results for SeqNet (S5) as per [Supp. Table III on Page 10](https://arxiv.org/pdf/2102.11603.pdf).
To obtain other results for the same table:
```python
# Raw Single (NetVLAD) Descriptor
python main.py --mode test --pooling single --dataset nordland-sf --seqL 1 --split test

# SeqNet (S1)
python main.py --mode test --pooling seqnet --dataset nordland-sf --seqL 1 --split test --resume ./data/runs/Jun02_00-07-14_l1_l1w1/

# Raw + Smoothing
python main.py --mode test --pooling smooth --dataset nordland-sf --seqL 5 --split test

# Raw + Delta
python main.py --mode test --pooling delta --dataset nordland-sf --seqL 5 --split test

# Raw + SeqMatch
python main.py --mode test --pooling single+seqmatch --dataset nordland-sf --seqL 5 --split test

# SeqNet (S1) + SeqMatch
python main.py --mode test --pooling s1+seqmatch --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun02_00-07-14_l1_l1w1/

# HVPR (S5 to S1)
# Run S5 first and save its predictions by specifying `resultsPath`
python main.py --mode test --pooling seqnet --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun02_00-08-23_l10_l10w5/ --resultsPath ./data/results/
# Now run S1 + SeqMatch using results from above (the timestamp of `predictionsFile` would be different in your case)
python main.py --mode test --pooling s1+seqmatch --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun02_00-07-14_l1_l1w1/ --predictionsFile ./data/results/Jun02_16-07-36_l5_0.npz

```

### Acknowledgement
The code in this repository is based on [Nanne/pytorch-NetVlad](https://github.com/Nanne/pytorch-NetVlad). Thanks to [Tobias Fischer](https://github.com/Tobias-Fischer) for his contributions to this work as a part of our project [QVPR/Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD).

### Citation
```
@article{garg2021seqnet,
  title={SeqNet: Learning Descriptors for Sequence-based Hierarchical Place Recognition},
  author={Garg, Sourav and Milford, Michael},
  journal={IEEE Robotics and Automation Letters (in press)},
  volume={6},
  number={3},
  pages={4305-4312},
  year={2021},
  publisher={IEEE},
  doi={10.1109/LRA.2021.3067633}
}
```

#### Other Related Projects
[Patch-NetVLAD (2021)](https://github.com/QVPR/Patch-NetVLAD);
[Delta Descriptors (2020)](https://github.com/oravus/DeltaDescriptors);
[CoarseHash (2020)](https://github.com/oravus/CoarseHash);
[seq2single (2019)](https://github.com/oravus/seq2single);
[LoST (2018)](https://github.com/oravus/lostX)

<a name="nordclean">[a]<a> This is the clean version of the dataset that excludes images from the tunnels and red lights, exact image names can be obtained from [here](https://github.com/QVPR/Patch-NetVLAD/blob/main/patchnetvlad/dataset_imagenames/nordland_imageNames_index.txt).
