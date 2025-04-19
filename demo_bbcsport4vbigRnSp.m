clear;
clc;

addpath(genpath('funs/'));
addpath(genpath('tSVD/'));

res = [];
[res_IMG]=[];
[res_PVC]=[];
[res_GPVCO]=[];

%% Load ORL dataset
f=1;
load("dataset\bbcsport4vbigRnSp.mat"); c=5;truth=truth';
load("dataset\bbcsport4vbigRnSp_percentDel_0.55.mat");
ind_folds = folds{f};
numClust = length(unique(truth));
num_view = length(X);
[numFold,numInst]=size(ind_folds);

result=[];

for iv = 1:num_view
    X1 = X{iv}';
    X1 = NormalizeFea(X1,1);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(ind_0,:) = [];
    Y{iv} = X1';
    W1 = eye(size(ind_folds,1));
    W1(ind_0,:) = [];
    G{iv} = W1;

end
%% Graph construction
S_temp=graph_construction(Y);
for i=1:num_view
    S(:,:,i)=G{i}'*S_temp{i}*G{i};
    [nu,~]=size(S_temp{i});
    omega(:,:,i)=G{i}'*ones(nu,nu)*G{i};
end
X = Y;
mu = 1e-4; 
rho = 1.1;
%% Training
for p= 0.2
    for lambda2 = 1e-1
        lambda3 = 1e-1;
        mode=2;
        [res] = My_comple1(X,S_temp,G,truth,c,omega,lambda2,lambda3,p,mode,mu,rho);
        [result]=[result;res];
        fprintf(fid,'p: %f ',p);
        fprintf(fid,'lambda2: %f ',lambda2);
        fprintf(fid,'res1: %g %g %g  \n',res(:,1:3));
    end
end