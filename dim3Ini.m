function dim3 = dim3Ini(datas, numClass, dataNum, tTime)

dim3.lam1 = sum(sum(sum(datas)))/(dataNum^2*tTime);
dim3.lam2 = 1-dim3.lam1;
dim3.nums = numClass*ones(1, dataNum);
dim3.dataNum = dataNum;
dim3.tTime = tTime;

seLabel = zeros(dataNum, dataNum, tTime+1);
reLabel = zeros(dataNum, dataNum, tTime+1);
ps = ones(1, numClass)/numClass;
for i = 1:dataNum
    for j = 1:dataNum
        seLabel(i,j,1) = 1+sum(rand>cumsum(ps));
        reLabel(i,j,1) = 1+sum(rand>cumsum(ps));
    end
end
for t = 2:(tTime+1)
    for i=1:dataNum
        re_value = mnrnd(1, 1/numClass*ones(1, numClass), dataNum);
        se_value = mnrnd(1, 1/numClass*ones(1, numClass), dataNum);        
        for j=1:dataNum
            reLabel(i,j, t) = find(re_value(j,:)==1);
            seLabel(i,j, t) = find(se_value(j,:)==1);
        end
    end
end

Nikt = zeros(numClass, numClass, dataNum);

for t = 2:(dim3.tTime+1)
    for i = 1:dataNum
        for j = 1:dataNum
            Nikt(seLabel(i,j,t-1), seLabel(i,j,t), i) = Nikt(seLabel(i,j,t-1), seLabel(i,j,t), i) + 1;
            Nikt(reLabel(i,j,t-1), reLabel(i,j,t), j) = Nikt(reLabel(i,j,t-1), reLabel(i,j,t), j) + 1;
        end
    end
end

dim3.kth_sem = [];
dim3.kth_rem = [];
dim3.reLabel = reLabel;
dim3.seLabel = seLabel;
dim3.Nikt = Nikt;
dim3.m_val = ones(1, numClass);