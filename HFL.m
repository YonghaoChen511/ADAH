function [cross_MAP, single_MAP, Ps, Pt] = HFL(B, dataset, param)
Xs = dataset.Xs';
Xt = dataset.Xt';
X_test = dataset.Xt_test';
lambda = param.lambda;
ns = dataset.ns;        
nt = dataset.nt;       
n = ns + nt;
Bs = B(1:ns,:);
Bt = B(ns+1:end,:);

Ps = pinv(Xs * Xs' + lambda * eye(size(Xs,1))) * (Xs * Bs);     % cross-domain
B_test = X_test' * Ps >= 0;        
B_te_comp = compactbit(B_test);
B_train = (Bs >= 0);                
B_tr_comp = compactbit(B_train);
Dhamm = hammingDist(B_te_comp, B_tr_comp);
[recall, precision, ~] = recall_precision(dataset.WTT, Dhamm);
cross_MAP = area_RP(recall, precision) * 100;

Pt = pinv(Xt * Xt' + lambda * eye(size(Xt,1))) * Xt * Bt;     % single-domain
B_test = X_test' * Pt >= 0;        
B_te_comp = compactbit(B_test);
B_train_single = (Bt >= 0);        
B_tr_comp_single = compactbit(B_train_single);
Dhamm = hammingDist(B_te_comp, B_tr_comp_single);
[recall, precision, ~] = recall_precision(dataset.WTT_single, Dhamm);
single_MAP = area_RP(recall, precision) * 100;

end