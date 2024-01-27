# EDIT: 27 Jan 2024
# Please download nordland-clean dataset from here: https://universityofadelaide.box.com/s/zkfk1akpbo5318fzqmtvlpp7030ex4up
# Please download nordland/oxford precomputed descriptors from here: https://universityofadelaide.box.com/s/p8uh5yncsaxk7g8lwr8pihnwkqbc2pkf
# Please download trained models from here: https://universityofadelaide.box.com/s/mp45yapl0j0by6aijf5kj8obt8ky0swk

# download nordland-clean dataset
#wget -cO - https://cloudstor.aarnet.edu.au/plus/s/PK98pDvLAesL1aL/download > nordland-clean.zip
#mkdir -p ./data/
#unzip nordland-clean.zip -d ./data/
#rm nordland-clean.zip

# download oxford descriptors
#wget -cO - https://cloudstor.aarnet.edu.au/plus/s/T0M1Ry4HXOAkkGz/download > oxford_2014-12-16-18-44-24_stereo_left.npy
#wget -cO - https://cloudstor.aarnet.edu.au/plus/s/vr21RnhMmOkW8S9/download > oxford_2015-03-17-11-08-44_stereo_left.npy
#mv oxford* ./data/descData/netvlad-pytorch/

# download trained models
#wget -cO - https://cloudstor.aarnet.edu.au/plus/s/oMwpOzex5ld4nQq/download > models-nordland.zip
#unzip models-nordland.zip -d ./data/
#rm models-nordland.zip
