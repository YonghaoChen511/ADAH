function B = ASPH(S, dataset, param)
warning off;

Xs = dataset.Xs';        % Դ������
ns = dataset.ns;        % Դ����������
Xt = dataset.Xt';        % Ŀ��������
nt = dataset.nt;        % Ŀ��������������ѵ������
n = ns + nt;

nbit = param.nbit;
maxIter = param.maxIter;

%% ��ʼ��
B = sign(randn(n, nbit)); 
B(B==0) = -1;      
V = B;

%% plot the convergence curve
if param.plot_loss_acc ==1 
    figure('Color','w');
    h = animatedline;
    h.Color = 'r';
    h.LineWidth = 1.3;
    h.LineStyle = '-.';
    title('L2');
end
%% 
gama = nbit;
omega = param.omega;
for t = 1:maxIter
    % update V
    Z = gama*(S*B)+omega*B;
    Temp = Z'*Z-1/n*(Z'*ones(n,1)*(ones(1,n)*Z));
    [~,Lmd,QQ] = svd(Temp); 
    idx = (diag(Lmd)>1e-6);
    Q = QQ(:,idx); 
    Q_ = orth(QQ(:,~idx));
    P = (Z-1/n*ones(n,1)*(ones(1,n)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
    P_ = orth(randn(n,nbit-length(find(idx==1))));
    V = sqrt(n)*[P P_]*[Q Q_]';

    % update B
    B = sign(gama*(S*V)+omega*V);
    
    % objective function value
    if param.plot_loss_acc == 1
        [loss(t)] = convergence(S, V, B, omega, nbit);
        addpoints(h, t, loss(t));
        drawnow;
    end
end
end

function [loss] = convergence(S, V, B, omega, nbit)
term1 = norm(B*V'-nbit*S, 'fro');
term2 = norm(B-V, 'fro');
loss = term1 + omega * term2;
end
