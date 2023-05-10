clear all;
close all;
warning off;

addpath('./utils/');

param.maxIter = 20;
param.plot_loss_acc = 0;      % plot convergence?

param.anchor = 17;            % MNIST-USPS
param.omega = 1;
param.lambda = 1e2;

% param.anchor = 52;            % Office-Home
% param.omega = 1e-4;
% param.lambda = 1e-1;
% 
% param.anchor = 39;            % Caltech256-ImageNet
% param.omega = 1;
% param.lambda = 1;


%% ---------------------------------------------------------------
nbits = [16 32 48 64 96 128];
param.ReducedDim = 256;        % MNIST-USPS is 256, others are 1024
test_num = 0.1;         % number of test set: 10% / 500

disp('starting');
for b = 1:length(nbits)
    mAP = [];
    param.nbit = nbits(b);
    for t = 1:10
        dataset = construct_dataset('MNIST-USPS', test_num, param);
        
        S = ABSR(dataset, param);               % Anchor-Based Similariity Reconstruction
        B = ASPH(S, dataset, param);            % Asymmetric Similarity Preserving Hashing
        [cross_MAP, single_MAP, Ps, Pt] = HFL(B, dataset, param);      % Hash Function Learning
        
        fprintf('nbit=%.0f, cross_MAP=%.2f, single_MAP=%.2f \n', param.nbit, cross_MAP, single_MAP);
        mAP(1, t) = cross_MAP;
        mAP(2, t) = single_MAP;
    end
    fprintf('nbit=%d \n', nbits(b));
    fprintf('%.2f \n', mean(mAP'));
    fprintf('%.2f \n', std(mAP'));
end
            