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
    if cc1==2%%����Rռ��
        PP=[PP;C(i,1:k+M)];%%������ѡ��
    elseif cc2==2%%R��Cռ��
        PP=[PP;R(i,1:k+M)];%%����������
    elseif cc1==1
        PP=[PP;R(i,1:k+M);C(i,1:k+M)];%%����ռ�ţ��Ͷ���������
    end
end

