# MccDTI
## Paper "Multi-view network embedding for drug-target Interactions prediction by consistent and complementary information preserving"

## Dependencies
MccDTI is tested on Ubuntu 18.04 with Python 3.6.2 and Matlab R2019a installed.

## Code
- DTI.m: predict drug-target interactions (DTIs) with PU-Matrix Completion
- run_DTINet.m: running DTI.m for drug-target prediction
- run_embedding.py: learning low-dimensional representation by multi-view network embedding model
- Database/preprocessing.py: preprocess the similarity data of drug and protein to obtain high-order similarity data

## Data
#### Database/deepDTnet_data/ directory: deepDTnet dataset (originally data from https://github.com/ChengF-Lab/deepDTnet)
- drug_node_list.txt: list of drug unique identifier and drug names and index
- drugdrug.txt: Drug-Drug interaction similarity
- drugDisease.txt: Drug-Disease association similarity
- drugsideEffect.txt: Drug-SideEffect association similarity
- drugsim1network.txt: Drug chemical similarity
- drugsim2network.txt: Drug therapeutic similarity
- drugsim3network.txt: Drug sequence similarity
- drugsim4network.txt: Drug biological processes similarity
- drugsim5network.txt: Drug cellular component similarity
- drugsim6network.txt: Drug molecular function similarity
- protein_node_list.txt: list of protein unique identifier and protein names and index
- proteinprotein.txt: Protein-Protein interaction similarity
- proteinDisease.txt: Protein-Disease association similarity
- proteinsim1network.txt: Protein sequence similarity
- proteinsim2network.txt: Protein biological processes similarity
- proteinsim3network.txt: Protein cellular component similarity
- proteinsim4network.txt: Protein molecular function similarity
- drugProtein.txt: Known 4,978 drug-target interactions connecting 732 approved drugs and 1,915 human targets.

#### Database/DTINet_data/ directory: DTINet dataset (originally data from https://github.com/luoyunan/DTINet)
- drug_node_list.txt: list of drug unique identifier and drug names and index
- Drugs.txt: Drug chemical similarity
- drug_drug.txt: Drug-Drug interaction similarity
- drug_disease.txt: Drug-Disease association similarity
- drug_se.txt: Drug-SideEffect association similarity
- protein_node_list.txt: list of protein unique identifier and protein names and index
- Proteins.txt: Protein sequence similarity
- protein_protein.txt: Protein-Protein interaction similarity
- protein_disease.txt: Protein-Disease association similarity
- mat_drug_protein.txt: Known 1,923 drug-target interactions connecting 708 approved drugs and 1,512 human targets.


## Results
#### Result/deepDTnet_data/ directory:
- Supplementary Data 1.csv: The list of top 3000 drug-target interactions predicted by MccDTI on deepDTnet dataset
- drug_emb_900.txt: Drug representation feature of deepDTnet dataset
- protein_emb_900.txt: Protein representation feature of deepDTnet dataset

#### Result/DTINet_data/ directory:
- Supplementary Data 2.csv: The list of top 3000 drug-target interactions predicted by MccDTI on DTINet dataset
- drug_emb_600.txt: Drug representation feature of DTINet dataset
- protein_emb_900.txt: Protein representation feature of DTINet dataset

## Tutorial
### Preprocess data
To get high-order similarity data, enter into the Database directory and run preprocessing script, e.g.
```
python preprocessing.py
```

### Learning low-dimensional representations of drug and protein
Using initial similarity and high-order similarity data and run the corresponding script, e.g.
```
python run_embedding.py
```

### DTIs Predication
Run the corresponding script, e.g.
```
matlab run_DTI.m
```

### Tips
Please create the following log folders in this project directory.
```
./Log/deepDTnet_data
./Log/DTINet_data
```

## Paper
Please cite our paper if you find the code useful for your research.
***
