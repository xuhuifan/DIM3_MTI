% clear all;
%% Dynamic Infinite Mixed-Membership Relational Model Sampling
clear;
% Niteration = 30000;
% %% data simulation
% dataNum = 40;
% tTime = 4;
% ininumClass = 3;
% [datas, ith_sela, ith_rela] = dataGens_1(dataNum, tTime);
% E = load('senator.mat');
% datas = E.E;
% [dataNum, ds, tTime] = size(datas);
%% initialization

load('enrondata.mat');
Niteration = 5;
dim3 = dim3Ini(datas,numClass, dataNum, tTime);
dim3.datas = datas;
dim3.kappa = 0.01;
dim3.gamma = 0.3;
dim3.alpha = ones(1, dataNum);
ite_numc = zeros(1, Niteration);
deviance_numc = zeros(1, Niteration);

st_like = -inf*ones(1, 5);
like_seL = dim3.seLabel;
like_reL = dim3.reLabel;
selec_like = zeros(1, 5);
cu_like = zeros(1, Niteration);
st_dims = cell(1, 5);


for n_ite = 1:Niteration
    % sampling \beta value
    dim3.betas = dirrnd([dim3.m_val dim3.gamma], 1);
    % sampling se_Labels re_Labels value
    dim3=label_gibbs(dim3);
     
    [dim3.deviance, cu_likes]= gibbs_dev(dim3);
    
    % sampling m value
    dim3 = m_stick(dim3);

    cu_like(n_ite) = cu_likes;
    
    spe_k = ceil(n_ite/Niteration*5);
    
    if cu_likes > st_like(spe_k)
        st_like(spe_k) = cu_likes;
        like_seL = dim3.seLabel;
        like_reL = dim3.reLabel;
        selec_like(spe_k) = n_ite;
        st_dims{spe_k} = dim3;
    end
    
    deviance_numc(n_ite) = dim3.deviance;
    ite_numc(n_ite) = max(dim3.nums);
    if mod(n_ite, 100)==0

        fprintf('the iteration time is %d\n', n_ite);
        fprintf('num of class is %d\n', max(dim3.nums));

    end
    
end

