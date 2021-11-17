clear;
clc
dim_drug = 600;
dim_prot = 900;
dim_imc = 50;

%interaction = load('./Database/deepDTnet_data/original_data/drugProtein.txt');
interaction = load('./Database/DTINet_data/mat_drug_protein.txt');

% load embedding features
%drug_feat = load(['./Result/deepDTnet_data/drug_emb_',num2str(dim_drug),'.txt']);
%prot_feat = load(['./Result/deepDTnet_data/protein_emb_',num2str(dim_prot),'.txt']);
drug_feat = load(['./Result/DTINet_data/drug_emb_',num2str(dim_drug),'.txt']);
prot_feat = load(['./Result/DTINet_data/protein_emb_',num2str(dim_prot),'.txt']);


nFold = 10;%deepDTnet 5, DTINet 10
Nrepeat = 5;


AUROC = zeros(Nrepeat, 1);
AUPRC = zeros(Nrepeat, 1);
re = [];

for p = 1 : Nrepeat
    fprintf('Repetition #%d\n', p);
    [AUROC(p), AUPRC(p), re{p}] = DTI(p, nFold, interaction, drug_feat, prot_feat, dim_imc);
end
[maxPR,index]=max(AUPRC);
prediction=re{1,index};
%dlmwrite(['./Result/deepDTnet_data/prediction_',num2str(dim_drug),'_',num2str(dim_prot),'.txt'], prediction, '\t')
dlmwrite(['./Result/DTINet_data/prediction_',num2str(dim_drug),'_',num2str(dim_prot),'.txt'], prediction, '\t')
for i = 1 : Nrepeat
	fprintf('Repetition #%d: AUROC=%.6f, AUPR=%.6f\n', i, AUROC(i), AUPRC(i));
end
fprintf('Mean: AUROC=%.6f, AUPR=%.6f\n', mean(AUROC), mean(AUPRC));