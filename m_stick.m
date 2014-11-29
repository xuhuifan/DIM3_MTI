function dim3 = m_stick( dim3 )
% sampling m's value
%  dim3 is the structure used

betas = dim3.betas;
alphas = dim3.alpha;
nums = dim3.nums;
seLabel = dim3.seLabel;
reLabel = dim3.reLabel;
Nikt = dim3.Nikt;
%%% need to remove empty clusters to adjust the right class number
empty_clu = [];
numClass = max(nums);

for k_m = 1:numClass  % searching the empty component
    if ((isempty(find(seLabel(:,:,2:(dim3.tTime+1)) == (k_m))))&(isempty(find(reLabel(:,:,2:(dim3.tTime+1)) == (k_m)))))
        empty_clu = [empty_clu k_m];
    end
end

kth_sem = [];
kth_rem = [];
if ~isempty(empty_clu)  % clean the empty component
    for k= 1:length(empty_clu)
        a1 = find(seLabel(:,:,1)==empty_clu(k));
        kth_sem = [kth_sem;a1];
        kth_rem = [kth_rem;find(reLabel(:,:,1)==empty_clu(k))];
    end
    
    for i = 1:dim3.dataNum
        if any(empty_clu <= nums(i))
            nums(i) = nums(i) - sum(empty_clu <= nums(i));
        end
    end
    
    for k = 1:numClass
        if any(empty_clu <= k)
            trans_k = k-sum(empty_clu <= k);
            seLabel(seLabel==k) = trans_k;
            reLabel(reLabel==k) = trans_k;
        end
    end
    
    numClass = max(nums);    
    Nikt(empty_clu, :, :) = [];
    Nikt(:, empty_clu, :) = [];
    betas(empty_clu) = [];
end


% if (2*3*dim3.dataNum^2-sum(sum(sum(Nikt))))~=(length(kth_sem)+length(kth_rem))
%    
%     a=1;
% end


% get the m_val value

m_val = zeros(1, numClass);
nohat_m = zeros(1, numClass);
for i_sli = 1:dim3.dataNum
    for l_sli = 1:nums(i_sli)
        nohat_t = zeros(1, nums(i_sli));
        t_Table = zeros(1, nums(i_sli));
        for k_sli = 1:nums(i_sli)
            i_max = Nikt(l_sli, k_sli, i_sli);
            if i_max > 0
                i_stir = stirling(i_max).*((alphas(i_sli)*betas(k_sli)).^(1:i_max));
                i_nt = 1+sum((rand*sum(i_stir)) > cumsum(i_stir));
                if (l_sli == k_sli)
                    i_p = dim3.kappa/(alphas(i_sli)*betas(k_sli)+dim3.kappa);
                    t_Table(k_sli) = i_nt-binornd(i_nt, i_p);
                else
                    t_Table(k_sli) = i_nt;
                end
                nohat_t(k_sli) = i_nt;
            end
        end
        nohat_m(1:nums(i_sli)) = nohat_m(1:nums(i_sli)) + nohat_t;
        m_val(1:nums(i_sli)) = m_val(1:nums(i_sli)) + t_Table;
    end
end

% disp(m_val);

dim3.seLabel = seLabel;
dim3.reLabel = reLabel;
dim3.nohat_m = nohat_m;
dim3.m_val = m_val;
dim3.nums = nums;
dim3.Nikt = Nikt;
dim3.kth_sem = kth_sem;
dim3.kth_rem = kth_rem;
end

