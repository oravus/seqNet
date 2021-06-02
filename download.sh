wget -cO - https://cloudstor.aarnet.edu.au/plus/s/PK98pDvLAesL1aL/download > nordland-clean.zip
mkdir -p ./data/
unzip nordland-clean.zip -d ./data/
rm nordland-clean.zip
