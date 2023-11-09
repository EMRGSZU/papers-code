function PP=fast_nondominated_sort(C,R,M,k)
PP=[];
F=[];
nR=size(R,1);
for i=1:nR
    cc1=0;
    cc2=0;
    for j=1:M
        if C(i,k+j)<=R(i,k+j)
            cc1=cc1+1;
        elseif C(i,k+j)>R(i,k+j)
            cc2=cc2+1;
        end
    end
    if cc1==2%%不被R占优
        PP=[PP;C(i,1:k+M)];%%保留候选解
    elseif cc2==2%%R把C占优
        PP=[PP;R(i,1:k+M)];%%保留父代解
    elseif cc1==1
        PP=[PP;R(i,1:k+M);C(i,1:k+M)];%%互不占优，就都保留下来
    end
end

