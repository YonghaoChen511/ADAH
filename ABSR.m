function [S] = ABSR(dataset, param)
Xs = dataset.Xs';        % 源域样本
ns = dataset.ns;        % 源域样本数量
Xt = dataset.Xt';        % 目标域样本
nt = dataset.nt;        % 目标域样本数量（训练集）
n = ns + nt;
c = dataset.c;          % classes
d = dataset.d;          % dimension

m = param.anchor;             % number of anchor

nbit = param.nbit;
maxIter = param.maxIter;

A = zeros(d, m);        
Zs = zeros(m, ns);
Zs(:,1:m) = eye(m);
Zt = zeros(m, nt);
Zt(:,1:m) = eye(m);

Ws = randn(d, d);
Wt = randn(d, d);

alpha = ones(1,2)/2;

%% plot the convergence curve
if param.plot_loss_acc ==1 
    figure('Color','w');
    h = animatedline;
    h.Color = 'g';
    h.LineWidth = 1.3;
    h.LineStyle = '-.';
    title('L1');
end
%% 

for t = 1:maxIter
    % update Ws
    Cs = Xs * Zs' * A';
    [Us, ~, Vs] = svd(Cs, 'econ');
    Ws = Us * Vs';
  
    % update Wt
    Ct = Xt * Zt' * A';
    [Ut, ~, Vt] = svd(Ct, 'econ');
    Wt = Ut * Vt';
    
    % update A
    temp = alpha(1)^2*Ws'* Xs * Zs' + alpha(2)^2*Wt'* Xt * Zt';
    [Ua, ~, Va] = svd(temp, 'econ');
    A = Ua * Va';

    % update Zs
    G = 2 * (alpha(1)^2) * eye(m);
    G = (G + G')/2;
    options=optimset('Algorithm','interior-point-convex','Display','off');
    parfor i = 1:ns
        ff = -2 * (alpha(1)^2) *Xs(:, i)'* Ws * A;
        Zs(:, i)=quadprog(G, ff', [], [], ones(1, m), 1, zeros(m,1), ones(m, 1), [], options);
    end
    
    % update Zt
    G = 2 * (alpha(2)^2) * eye(m);
    G = (G + G')/2;
    options=optimset('Algorithm','interior-point-convex','Display','off');
    parfor j = 1:nt
        ff = -2 * (alpha(2)^2)*Xt(:, j)'* Wt * A;
        Zt(:, j)=quadprog(G, ff', [], [], ones(1, m), 1, zeros(m,1), ones(m, 1), [], options);
    end
    
    % update alpha
    alpha(1) = (norm(Xs-Ws*A*Zs,'fro')^2)^-1;
    alpha(2) = (norm(Xt-Wt*A*Zt,'fro')^2)^-1;
    Q = 1/sum(alpha);
    alpha = Q*alpha;
    
    % loss value
    if param.plot_loss_acc ==1 
        obj(t)=alpha(1)^2*norm(Xs-Ws*A*Zs,'fro') + alpha(2)^2*norm(Xt-Wt*A*Zt,'fro');
        addpoints(h, t, obj(t));
        drawnow;
    end
end

% similarity matrix reconstruction
Sss = Zs' * diag(1 ./ sum(Zs, 2))*diag(1 ./ sum(Zs, 2))' * Zs;
Stt = Zt' * diag(1 ./ sum(Zt, 2))* diag(1 ./ sum(Zt, 2))' * Zt;
Sst = Zs' * diag(1./sum(Zs, 2))* diag(1./sum(Zt, 2))' * Zt;
Sts = Zt' * diag(1./sum(Zt, 2))*diag(1./sum(Zs, 2))' * Zs;
S = [[Sss, Sst]; [Sts, Stt]];

S = S./repmat(sqrt(sum(S.^2,2)),1,size(S,2));       % 行规范化

end
