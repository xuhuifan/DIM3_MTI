function dim3 = label_gibbs( dim3 )
% Sampling re_label, se_label
se_Labels = dim3.seLabel;
re_Labels = dim3.reLabel;
nums = dim3.nums;

betas = dim3.betas;
alphas = dim3.alpha;
dataNum = dim3.dataNum;
kappas = dim3.kappa;
Nikt = dim3.Nikt;
kth_sem = dim3.kth_sem;
kth_rem = dim3.kth_rem;

% the formal sampler of se_Labels, re_Labels
numClass = max(nums);

for i=randperm(dim3.dataNum)
    for j=randperm(dim3.dataNum)
        if ~any((i+(j-1)*dim3.dataNum)==kth_sem)
            Nikt(se_Labels(i,j, 1), se_Labels(i,j,2), i) = Nikt(se_Labels(i,j, 1), se_Labels(i,j,2), i)-1;
        end
        if ~any((i+(j-1)*dim3.dataNum)==kth_rem)
            Nikt(re_Labels(i,j, 1), re_Labels(i,j,2), j) = Nikt(re_Labels(i,j, 1), re_Labels(i,j,2), j)-1;
        end
%         if sum(sum(sum(Nikt<0)))>0
%             
%             
%             a = 1;
%         end
        ns_weight = (alphas(i)*betas(se_Labels(i,j,2))+[Nikt(1:nums(i), se_Labels(i,j,2), i)]'+(kappas)*[(se_Labels(i,j,2) == (1:nums(i)))])./...
            (alphas(i)+[sum(Nikt(1:nums(i), 1:nums(i), i), 2)]'+kappas);
        nr_weight = (alphas(j)*betas(re_Labels(i,j,2))+[Nikt(1:nums(j), re_Labels(i,j,2), j)]'+(kappas)*[(re_Labels(i,j,2) == (1:nums(j)))])./...
            (alphas(j)+[sum(Nikt(1:nums(j), 1:nums(j), j), 2)]'+kappas);
        p_weight = ns_weight'*nr_weight;
        p_weight = reshape(p_weight, 1, []);
        
        % sampling
        ath_value = 1+sum(rand*sum(p_weight) > cumsum(p_weight));
        ath_value = min(ath_value, (nums(i)*nums(j)));
        ath_col = ceil(ath_value/(nums(i)));
        ath_row = ath_value - (ath_col-1)*(nums(i));
        se_Labels(i,j,1) = (ath_row);
        re_Labels(i,j,1) = (ath_col);
        
        Nikt(se_Labels(i,j, 1), se_Labels(i,j,2), i) = Nikt(se_Labels(i,j, 1), se_Labels(i,j,2), i)+1;
        Nikt(re_Labels(i,j, 1), re_Labels(i,j,2), j) = Nikt(re_Labels(i,j, 1), re_Labels(i,j,2), j)+1;
    end
end

% if any((sum(sum(Nikt, 1), 2))~=120)
%    
%     a = 2;
% end


tau_kl = zeros(numClass, numClass);
tau1_kl = zeros(numClass, numClass);

for t = 1:(dim3.tTime)
    for k = 1:numClass
        for l=1:numClass
            [x_loc, y_loc]=find((se_Labels(:,:,(t+1))==(k))&(re_Labels(:,:,(t+1))==(l)));
            tau1_kl(k,l)=tau1_kl(k,l)+sum(diag(dim3.datas(x_loc, y_loc, (t))));
            tau_kl(k,l) = tau_kl(k,l)+length(x_loc);
        end
    end
end

for t=2:(dim3.tTime+1)
    % calculate likelihood information
    for i=randperm(dim3.dataNum)
        for j=randperm(dim3.dataNum)
            
            se_la = se_Labels(i,j,t);
            re_la = re_Labels(i,j,t);
            
            Nikt(se_Labels(i,j,t-1), se_Labels(i,j,t), i) = Nikt(se_Labels(i,j,t-1), se_Labels(i,j,t), i)-1;
            Nikt(re_Labels(i,j,t-1), re_Labels(i,j,t), j) = Nikt(re_Labels(i,j,t-1), re_Labels(i,j,t), j)-1;
            if (t<(dim3.tTime+1))
                Nikt(se_Labels(i,j,t), se_Labels(i,j,t+1), i) = Nikt(se_Labels(i,j,t), se_Labels(i,j,t+1), i)-1;
                Nikt(re_Labels(i,j,t), re_Labels(i,j,t+1), j) = Nikt(re_Labels(i,j,t), re_Labels(i,j,t+1), j)-1;
            end
%         if sum(sum(sum(Nikt<0)))>0
%             
%             
%             a = 1;
%         end            
            
            if t<(dim3.tTime+1)
                ns1_weight = (alphas(i)*betas(se_Labels(i,j,t+1))+[Nikt(1:nums(i), se_Labels(i,j,t+1), i);0]'+kappas*[(se_Labels(i,j,t+1) == (1:nums(i))) 0]+...
                    [(se_Labels(i,j,t-1) == (1:nums(i))) 0].*[(se_Labels(i,j,t+1) == (1:nums(i))) 0])./...
                    (alphas(i)+[sum(Nikt(1:nums(i), 1:nums(i), i), 2);0]'+kappas+[(se_Labels(i,j,t-1) == 1:nums(i)) 0]);
                nr1_weight = (alphas(j)*betas(re_Labels(i,j,t+1))+[Nikt(1:nums(j), re_Labels(i,j,t+1), j);0]'+kappas*[(re_Labels(i,j,t+1) == (1:nums(j))) 0]+...
                    [(re_Labels(i,j,t-1) == (1:nums(j))) 0].*[(re_Labels(i,j,t+1) == (1:nums(j))) 0])./...
                    (alphas(j)+[sum(Nikt(1:nums(j), 1:nums(j), j), 2);0]'+kappas+[(re_Labels(i,j,t-1) == 1:nums(j)) 0]);
            else
                ns1_weight = ones(1, nums(i) + 1);
                nr1_weight = ones(1, nums(j) + 1);      
            end
            
            ns2_weight = alphas(i)*[betas(1:nums(i)) 1-sum(betas(1:nums(i)))]+[Nikt(se_Labels(i,j,t-1), 1:nums(i), i) 0]+kappas*[(se_Labels(i,j,t-1) == 1:nums(i)) 0];
            nr2_weight = alphas(j)*[betas(1:nums(j)) 1-sum(betas(1:nums(j)))]+[Nikt(re_Labels(i,j,t-1), 1:nums(j), j) 0]+kappas*[(re_Labels(i,j,t-1) == 1:nums(j)) 0];
            %             ns2_weight = (alphas(i)*[betas(1:nums(i)) 1-sum(betas(1:nums(i)))]+[Nikt(se_Labels(i,j,t-1), 1:nums(i), i) 0]+kappas*[(se_Labels(i,j,t-1) == 1:nums(i)) 0])./...
            %                 (alphas(i)*[betas(1:nums(i)) 1-sum(betas(1:nums(i)))]+[sum(Nikt(:, 1:nums(i), i)) 0]+kappas);
            %             nr2_weight = (alphas(j)*[betas(1:nums(j)) 1-sum(betas(1:nums(j)))]+[Nikt(re_Labels(i,j,t-1), 1:nums(j), j) 0]+kappas*[(re_Labels(i,j,t-1) == 1:nums(j)) 0])./...
            %                 (alphas(j)*[betas(1:nums(j)) 1-sum(betas(1:nums(j)))]+[sum(Nikt(:, 1:nums(j), j)) 0]+kappas);
            
            ns_weight = ns1_weight.*ns2_weight;
            nr_weight = nr1_weight.*nr2_weight;
            
            % edge(i,j,t)'s likelihood calculation
            tau_kl(se_la, re_la)=tau_kl(se_la, re_la)-1;
            tau1_kl(se_la, re_la)= tau1_kl(se_la, re_la)-dim3.datas(i,j,t-1);
            tau0_kl = tau_kl-tau1_kl;
            
            % calculating the likehood value
                        if (dim3.datas(i,j,t-1)==1)
                            like_wei = ([tau1_kl(1:nums(i),1:nums(j)) zeros(nums(i),1);zeros(1,(nums(j)+1))]+dim3.lam1)./([tau_kl(1:nums(i), 1:nums(j)) zeros(nums(i),1);zeros(1,(nums(j)+1))]+dim3.lam1+dim3.lam2);  % change the denominator
                        else
                            like_wei = ([tau0_kl(1:nums(i),1:nums(j)) zeros(nums(i),1);zeros(1,(nums(j)+1))]+dim3.lam2)./([tau_kl(1:nums(i), 1:nums(j)) zeros(nums(i),1);zeros(1,(nums(j)+1))]+dim3.lam1+dim3.lam2);
                        end
%             like_wei = ones(nums(i)+1, nums(j)+1);
            p_weight = diag(ns_weight)*like_wei*diag(nr_weight);
            
            p_weight = reshape(p_weight, 1, []);
            
            % sampling
            ath_value = 1+sum(rand*sum(p_weight) > cumsum(p_weight));
            %             if ath_value > ((nums(i)+1)*(nums(j)+1))
            %                 ath_value = ((nums(i)+1)*(nums(j)+1));
            %             end
            ath_value = min(ath_value, ((nums(i)+1)*(nums(j)+1)));
            
            ath_col = ceil(ath_value/(nums(i)+1));
            ath_row = ath_value - (ath_col-1)*(nums(i)+1);
            
            % consider the case (ath_col > numClass) or (ath_row > numClass)
            if ((ath_row > numClass)||(ath_col > numClass))
                bb = dirrnd([1 dim3.gamma], 1);
                betas(end:(end+1)) = betas(end)*bb;
                tau_kl = [tau_kl zeros(numClass, 1); zeros(1, (numClass+1))];
                tau1_kl = [tau1_kl zeros(numClass, 1); zeros(1, (numClass+1))];
                Nikt = [Nikt zeros(numClass, 1, dataNum)];
                Nikt = [Nikt;zeros(1, numClass+1, dataNum)];
                numClass = numClass+1;
            end
            if ath_row > nums(i)  % here is something wrong
                nums(i) = nums(i)+1;
            end
            if  ath_col > nums(j)
                nums(j) = nums(j)+1;
            end
            
            se_Labels(i,j,t) = (ath_row);
            re_Labels(i,j,t) = (ath_col);
            
            tau_kl(ath_row, ath_col)=tau_kl(ath_row, ath_col)+1;
            tau1_kl(ath_row, ath_col)=tau1_kl(ath_row, ath_col)+dim3.datas(i,j,t-1);
            
            % we need to increase Nikt's specific value
            Nikt(se_Labels(i,j,t-1), se_Labels(i,j,t), i) = Nikt(se_Labels(i,j,t-1), se_Labels(i,j,t), i)+1;
            Nikt(re_Labels(i,j,t-1), re_Labels(i,j,t), j) = Nikt(re_Labels(i,j,t-1), re_Labels(i,j,t), j)+1;
            if (t<(dim3.tTime+1))
                Nikt(se_Labels(i,j,t), se_Labels(i,j,t+1), i) = Nikt(se_Labels(i,j,t), se_Labels(i,j,t+1), i)+1;
                Nikt(re_Labels(i,j,t), re_Labels(i,j,t+1), j) = Nikt(re_Labels(i,j,t), re_Labels(i,j,t+1), j)+1;
            end
            
        end
    end
    
end



%%% 
dim3.seLabel = se_Labels;
dim3.reLabel = re_Labels;
dim3.betas = betas;
dim3.nums = nums;
dim3.Nikt = Nikt;
end

