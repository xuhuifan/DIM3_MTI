function [deviance, li_jps] = gibbs_dev(dim3)
deviance = 0;
se_Labels = dim3.seLabel;
re_Labels = dim3.reLabel;
nums = dim3.nums;

Nikt = dim3.Nikt;
li_jps = 0;
numClass = max(nums);

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

for t_dev = 2:(dim3.tTime+1)
 
    for i_dev = 1:dim3.dataNum
        for j_dev = 1:dim3.dataNum
            seL = (se_Labels(i_dev, j_dev, t_dev));
            reL = (re_Labels(i_dev, j_dev, t_dev));
            
            tau_kl(seL, reL)=tau_kl(seL, reL)-1;
            tau1_kl(seL, reL)= tau1_kl(seL, reL)-dim3.datas(i_dev,j_dev,t_dev-1);
            tau0_kl = tau_kl-tau1_kl;
            
            %             if sum(sum(tau_kl < 0)) >0
            %                 fprintf('here here \n');
            %             end
            %             if sum(sum(tau1_kl < 0)) >0
            %                 fprintf('here here \n');
            %             end
            % deviance's calculation
            if (dim3.datas(i_dev,j_dev,t_dev-1)==1)
                like_wei = (tau1_kl+dim3.lam1)./(tau_kl+dim3.lam1+dim3.lam2);
            else
                like_wei = (tau0_kl+dim3.lam2)./(tau_kl+dim3.lam1+dim3.lam2);
            end
            wei_ij = (diag(Nikt(se_Labels(i_dev, j_dev, t_dev-1), :, i_dev))*like_wei)*diag(Nikt(re_Labels(i_dev, j_dev, t_dev-1), :, j_dev))/(4*(dim3.dataNum)^2*dim3.tTime);
            deviance = deviance+log(sum(sum(wei_ij)));
            % end
           
            if (dim3.datas(i_dev,j_dev,t_dev-1)==1)
                like_wei = (tau1_kl(seL, reL)+dim3.lam1)/(tau_kl(seL, reL)+dim3.lam1+dim3.lam2);
            else
                like_wei = (tau0_kl(seL, reL)+dim3.lam2)/(tau_kl(seL, reL)+dim3.lam1+dim3.lam2);
            end
            
%             ratio_l = min([log(like_wei)/log(ps), log(like_wei)/log(pr)]);
%             if ratio_l < 1
%                 ratio_time = ratio_time + 1;
%             end
%             ratio_u = max([ratio_u, log(like_wei)/log(ps), log(like_wei)/log(pr)]);
            li_jps = li_jps + log(like_wei);
%             if i_dev == j_dev
%                 Nikt(i_dev,:) = Nikt(i_dev,:)-1;
%             end
            
            tau_kl(seL, reL)=tau_kl(seL, reL)+1;
            tau1_kl(seL, reL)= tau1_kl(seL, reL)+dim3.datas(i_dev,j_dev,t_dev-1);
        end
    end
end
deviance = -2*deviance;
end