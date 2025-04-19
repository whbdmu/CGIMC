function res=My_comple1(X,S,G,truth,c,omega,lambda2,lambda3,p,mode,mu,rho)
lambda1 = 0;
%% Initialization
[~,n2,~]=size(S);
[~,l] = size(G{1});
sum_G = 0;
Q = zeros(c,l);
Y1 = Q;
R = Q*Q';

for i=1:n2 %n2 = iv
    S_sss(:,:,i)=G{i}'*S{i}*G{i};
    Q2{i}=zeros(size(S{i}));
    P{i}=Q2{i};
    E{i}=P{i};
    sum_G = sum_G + G{i}'*G{i};
    U{i} = X{i}*(Q*G{i}')';
end

dim=size(S_sss);
[n1,~,n3]=size(S_sss);
p1 = ones(n1, 1);
tol = 1e-8; max_iter = 400; 
Q1=zeros(dim); Q3=zeros(dim); W=zeros(dim); Y=zeros(dim);
M=zeros(dim);
iter=0;



Z=S_sss;%Z = S n*n
beta=ones(n1,1);

for i=1:n3
    [nu,~]=size(S{i});%nu = que shi
    omega(:,:,i)=G{i}'*ones(nu,nu)*G{i};%ones(numFold,numFold);
end
for i=1:max_iter
    iter=iter+1;
    X_k=Z;
    Z_k=M;
    Sum_Z = 0;
    for j=1:n3
        P{j}=S{j}+1/mu*Q2{j}-E{j};
        Sum_Z = Sum_Z + Z(:,:,j);
    end


    %% Update Z
    for j=1:n3
        Z(:,:,j)=0.5*(M(:,:,j)-1/mu*Q1(:,:,j)+W(:,:,j)-1/mu*Q3(:,:,j))-1/mu*(Y(:,:,j).*omega(:,:,j));
    end
    
    %% Update  W
    for j=1:n3
        temp1 = L2_distance_1(Q,Q);
        temp2 = Z(:,:,j)+Q3(:,:,j)/mu;
        linshi_W = temp2-lambda2*temp1/mu;
        linshi_W = linshi_W-diag(diag(linshi_W));
        for ic = 1:size(Z(:,:,j),2)
            ind = 1:size(Z(:,:,j),2);
            %             ind(ic) = [];
            W(ic,ind,j) = EProjSimplex_new(linshi_W(ic,ind));
        end
    end
    clear temp1 temp2


       %% Update  R
   temp = (Q*Y1')';
   temp(isnan(temp)) = 0;
   temp(isinf(temp)) = 1e10;
   [Gs,~,Vs] = svd(temp,'econ');
   Gs(isnan(Gs)) = 0;
   Vs(isnan(Vs)) = 0;
   R = Gs*Vs';
   clear Gs Vs temp

%     [tmp_u, ~, tmp_v] = svd(Q*Y1');
%      R = tmp_v * tmp_u';


    %% Update M
    [M,~,~] = prox_tnn(Z+Q1/mu,beta/mu,p,mode);
    
    %% Update E
    for j=1:n3
        temp1 = S{j}-P{j}+Q2{j}/mu;
        temp2 = lambda1/mu;
        E{j}= max(0,temp1-temp2)+min(0,temp1+temp2);
    end
    
    clear temp1 temp2

      %% Update U
    for iv = 1:n3
        temp = X{iv}*(Q*G{iv}')';
        temp(isnan(temp)) = 0;
        temp(isinf(temp)) = 1e10;
        [Gs,~,Vs] = svd(temp,'econ');
        Gs(isnan(Gs)) = 0;
        Vs(isnan(Vs)) = 0;
        U{iv} = Gs*Vs';
        clear Gs Vs
    end




    %% Update Q
    UXG = 0;
    L_W = 0;
    for iv = 1:n3
        WW = (W(:,:,iv)+W(:,:,iv)')*0.5;
        LW = diag(sum(WW))-WW;
        UXG = U{iv}'*X{iv}*G{iv}+UXG;
        L_W = L_W+LW;
    end
    L_W = lambda2*L_W;
    H = sum_G + L_W+0.00000001*eye(size(L_W));
    Q = (UXG + lambda3*(R*Y1)/(H+lambda3*(eye(size(sum_G)))));




    %% Update Y
    sum_pFR = zeros(size(Y1));
    sum_pFR = R*Q;
    [~, y_idx] = max(sum_pFR', [], 2);
    Y1 = full(sparse(1:n1, y_idx, ones(n1, 1), n1, c));
    Y1 = Y1';
    
    %% Checking Convergence
    chgX=max(abs(Z(:)-X_k(:)));
    chgZ=max(abs(M(:)-Z_k(:)));
    chgX_Z=max(abs(Z(:)-M(:)));
    chg=max([chgX chgZ chgX_Z]);
    
%     if iter == 1 || mod(iter, 10) == 0
%         disp(['iter ' num2str(iter) ', mu = ' num2str(mu) ', chg = ' num2str(chg) ', chgX = ' num2str(chgX) ', chgZ = ' num2str(chgZ) ',chgX_Z = ' num2str(chgX_Z) ]);
%     end
    
%     if chg<tol
%         break;
%     end
    %% Update Lagrange multiplier
    for j=1:n3
        tP(:,:,j)=G{j}'*P{j}*G{j};
    end
    Q1=Q1+mu*(Z-M);
    for j=1:n3
        Q2{j}=Q2{j}+mu*(S{j}-P{j}-E{j});
    end
    Q3=Q3+mu*(Z-W);
    Y=Y+mu*(Z-tP).*omega;
    mu=min(rho*mu,1e10);


    %% Clustering
    newF = Y1';

    repeat = 5;
    for iter_c = 1:repeat
        %         pre_labels    = kmeans(real(new_F),c,'emptyaction','singleton','replicates',20,'display','off');
        [~, preY] = max(newF, [], 2);
        result_LatLRR = ClusteringMeasure(truth, preY);
        AC(iter_c)    = result_LatLRR(1)*100;
        MIhat(iter_c) = result_LatLRR(2)*100;
        Purity(iter_c)= result_LatLRR(3)*100;
    end
    mean_ACC = mean(AC);
    mean_NMI = mean(MIhat);
    mean_PUR = mean(Purity);
    res=[mean_ACC mean_NMI mean_PUR];
    if iter == 1 || mod(iter, 1) == 0
        disp(['iter ' num2str(iter) ', mean_ACC = ' num2str(mean_ACC) ', mean_NMI = ' num2str(mean_NMI) ', mean_PUR = ' num2str(mean_PUR)]);
        disp('---------------------------------------------------------------------------------------')
    end

end
