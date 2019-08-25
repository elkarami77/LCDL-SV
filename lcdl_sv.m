function [D,Z,U,b,L,class_list]=lcdl_sv(training_data,training_label,Dinit,...
    param,tol,VERBOSE)
%this function solves the objective function of locality constraint dictionary
% learning with support vector discriminative term (LCDL-SV)

%*********INPUT:************************************
%training_data:   m x n matrix containing n training samples with a dimension of m
%training_label:  1 x n vector containing the labels of n training samples
%Dinit:           initialized dictionary
%param:           parameters for the proposed LCDL-SV
%lambda1:         a trade-off coefficient for l2 regularizer
%lambda2:         a trade-off coefficient for discrimination term
%theta:           a trade-off coefficient for hinge loss function
%maxiter:         maximum number of iterations of the algorithm
% tol:             stopping criterion, minimal difference in norm between dictionary Dk
%                 (dictionary of current iteration) and Dkm1 (of previous iteration)
%VERBOSE:         controlls the amount of output printed on the screen
%*********OUTPUT:************************************
%D:               dictionary learned by the algorithm
%Z:               K x n matrix, each column is a coding vector
%U:               m x c matrix, c is the class number and each column is a discriminant function vector
%b                1 x c containing the bias term
%L:               the graph Laplacian matrix
%class_list       1 x c vector containing the list of class names

%Many thanks to Sijia Cai for releasing their codes for SVGDL.

if nargin < 5 || isempty(tol)
    tol = 1e-3;
end

if nargin < 6 || isempty(VERBOSE)
    VERBOSE = 0; %prints the relative error obtained in each iteration
end

lambda1 = param.lambda1;
lambda2 = param.lambda2;
theta = param.theta;
maxiter = param.maxiter;

tau = 1/theta;
X = training_data;
y = training_label;

m = size(X,1);
n = length(y);
K = size(Dinit,2);

class_list  = unique(y,'stable');
class_num   = length(class_list);
class_space = 1;
class_idx   = zeros(n, 1);

% define the label matrix Y_label for c two-class classification problems (one-vs-all)
Y = zeros(n, class_num);
for i = 1 : n
    for j = 1 : class_space
        if y(i) == class_list(j)
            class_idx(i) = j;
        end
    end
    if class_idx(i) == 0
        class_space = class_space + 1;
        class_idx(i) = class_space;
    end
    Y(i, class_idx(i)) = 1;
end

Y_label = sign(Y-0.5);

% initialize Z U b
Zinit = zeros(K,n);
Uinit = zeros(K,class_num);
binit = zeros(1,class_num);

k = 0;
rel_deltaD = 1;

Zk = Zinit;
Uk = Uinit;
bk = binit;
Dk = Dinit;

% parameters to construct the graph Laplacian matrix
options = [];
options.NeighborMode = 'KNN';
options.k = param.knn;
options.WeightMode = 'HeatKernel';

while k < maxiter && rel_deltaD > tol
    k = k+1;
    fprintf('\nIteration: %i, ', k);
    
    % update L
    Lk = construct_L(Dk,options);
    
    % update Zk
    Pk = inv(Dk'*Dk+lambda1*Lk);
    
    if k~=1
        for i = 1 : n
            Y_labelki = Zk(:,i)'*Uk + bk;
            loss_idx  = find( Y_labelki.*Y_label(i,:) < 1 ) ;
            if isempty(loss_idx)
                Zk(:,i) = Pk*Dk'*X(:,i);
            else
                Yi_idx = Y_label(i,loss_idx);
                Uk_idx = Uk(:,loss_idx);
                bk_idx = bk(loss_idx);
                
                ski = Dk'*X(:,i)+2*lambda2*theta*(Uk_idx*Yi_idx'-Uk_idx*bk_idx');
                Tki = inv(eye(size(Uk_idx,2))+2*lambda2*theta*Uk_idx'*Pk*Uk_idx);
                
                Zk(:,i)=(Pk-2*lambda2*theta*Pk*Uk_idx*Tki*Uk_idx'*Pk)*ski;
            end
        end
    else
        Zk = Pk*Dk'*X;
    end
    
    %update Dk
    Dkm1=Dk;
    
    D1 = Zk*Zk';
    [numd1,ad1]=size(find(isnan(D1)));
    [numd2,ad2]= size(find(isinf(D1)));
    if(numd1>0 )||(numd2>0)
        D1=eye(size(X,1));
    end
    D3 = pinv(D1);
    D2 = X*Zk';
    Dk = D2*D3;
    
    rel_deltaD = norm(Dk(:)-Dkm1(:))/norm(Dk(:));
    
    if  VERBOSE
        fprintf('\b  relative change of D = %g', rel_deltaD);
    end
    
    %update Uk bk
    [Uk, bk, ~] = li2nsvm_multiclass_lbfgs(Zk',y, tau);
end
D = Dk;
Z = Zk;
U = Uk;
b = bk;
L = Lk;
end