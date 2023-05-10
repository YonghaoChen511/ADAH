function [km,kd] = gaussiansimilarity(interaction,nl,nd )
%A: Binary relations between disease and miRNA, 1st column:miRNA, 2nd column:disease

%calculate gamad for Gaussian kernel calculation
 gamad = nl/(norm(interaction,'fro')^2);

%calculate Gaussian kernel for the similarity between disease: kd
C=interaction;
km=zeros(nl,nl);
D=C*C';
for i=1:nl 
    for j=i:nl
        km(i,j)=exp(-gamad*(D(i,i)+D(j,j)-2*D(i,j)));
    end
end
km=km+km'-diag(diag(km));
%calculate gamam for Gaussian kernel calculation

gamam = nd/(norm(interaction,'fro')^2);
%calculate Gaussian kernel for the similarity between miRNA: km
kd=zeros(nd,nd);
E=C'*C;
for i=1:nd
    for j=i:nd
        kd(i,j)=exp(-gamam*(E(i,i)+E(j,j)-2*E(i,j)));
    end
end
kd=kd+kd'-diag(diag(kd));
end