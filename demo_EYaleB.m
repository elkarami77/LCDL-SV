clear
clc
close all

%% load toolkits
addpath('.\large_scale_svm');

%% load data
load YaleB_DR_DAT
num_atom_per_class = 10; % number of atoms per class

tr_dat = Train_DAT;
tt_dat = Test_DAT;
trls = trainlabels;
ttls = testlabels;

clear Train_DAT Test_DAT trainlabels testlabels;

% reduce the dimension via PCA
rdim = 300;

Vt = Eigenface_f(tr_dat,rdim);
tr_dat = Vt'*tr_dat;
tt_dat = Vt'*tt_dat;
tr_dat = tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [size(tr_dat,1),1]) );
tt_dat = tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [size(tt_dat,1),1]) );

%% set parameters
param = [];
param.lambda1    =   1e-3;
param.lambda2    =   1e-6;
param.maxiter   =   20;
param.theta      =   5;
param.knn = 1;

%% initialize sub-dictionaries via pca
fprintf('\n------------------------Initializing Dictionary------------------------\n\n');
Dini = [];
num_class   = length(unique(trls));
num_atom_ci = num_atom_per_class;
fprintf('class:');
for ci = 1:num_class
    tr_dat_ci           =    tr_dat(:,trls==ci);
    [Dini_ci,~,mean_ci] =    Eigenface_f(tr_dat_ci,num_atom_ci-1);
    Dini_ci             =    [Dini_ci mean_ci./norm(mean_ci)];
    Dini                =    [Dini Dini_ci];
    fprintf('%d', ci);
    if ~mod(ci, 20)
        fprintf('.\n      ');
    else
        fprintf('.');
    end
end
fprintf('\n\nInitialization Is Done!')
D_label = repelem(1:num_class,num_atom_ci);

%% run algorithm
fprintf('\n\n----------------------------Algorithm LCDL-SV----------------------------\n\n');
[D,Z,U,b,L,class_list]  = lcdl_sv(tr_dat,trls,Dini,param);
fprintf('\n\nLCDL-SV Model Training Is Completed!')

%% encode the testing data
fprintf('\n\n--------------------------------Testing--------------------------------\n\n');
eta1 = 1e-2;
P = (D'*D+eta1*eye(size(D,2)))\D';

% parameter for fusing
eta2 = 5;

fuse_pred = zeros(1,size(tt_dat,2)); %predicted labels for LCDV-SV
res_pred = zeros(1,size(tt_dat,2)); %predicted labels for LCDV-SV (Res)
svm_pred = zeros(1,size(tt_dat,2)); %predicted labels for LCDV-SV (SVM)
for indTest = 1:size(tt_dat,2)
    y    =  tt_dat(:,indTest);
    coef = P*y;
    
    Y = coef'*U + repmat(b,[size(coef',1),1]);
    
    error = zeros(1,num_class);
    temp = zeros(1,num_class);
    for ci = 1:num_class
        coef_c   =  coef(D_label==ci);
        Dc       =  D(:,D_label==ci);
        error(ci) = norm(y-Dc*coef_c,2);
        temp(ci) = norm(coef_c);
    end
    reg_error = error./temp;
    
    score = reg_error-eta2*Y;
    index      =  find(score==min(score));
    id         =  index(1);
    fuse_pred(indTest) = id;
    
    [~,ind] = min(reg_error);
    res_pred(indTest) = ind;
    
    [~,ind] = max(Y);
    svm_pred(indTest) = ind;
    
end

% Output the recognition accuracy
fuse_acc  =  (sum(fuse_pred==ttls))/length(ttls);
res_acc  =  (sum(res_pred==ttls))/length(ttls);
svm_acc  =  (sum(svm_pred==ttls))/length(ttls);
fprintf('Accuracy of LCDL-SV (Res) is %.2f%%\n',res_acc*100)
fprintf('Accuracy of LCDL-SV (SVM) is %.2f%%\n',svm_acc*100)
fprintf('Accuracy of LCDL-SV is %.2f%%\n',fuse_acc*100)
