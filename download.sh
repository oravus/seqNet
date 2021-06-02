# download nordland-clean dataset
wget -cO - https://cloudstor.aarnet.edu.au/plus/s/PK98pDvLAesL1aL/download > nordland-clean.zip
mkdir -p ./data/
unzip nordland-clean.zip -d ./data/
rm nordland-clean.zip

# download trained models
wget -cO - https://cloudstor.aarnet.edu.au/plus/s/oMwpOzex5ld4nQq/download > models-nordland.zip
unzip models-nordland.zip -d ./data/
rm models-nordland.zip
