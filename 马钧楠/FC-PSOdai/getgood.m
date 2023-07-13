clear
load('EPI.mat')
load('matlab1.mat')
load('matlab2.mat')


for i=1:10
    for j=1:size(EP{1,i},1)
% [x(i,:),EP{i}]=PotentialPso(150,traind,cut); %��������ѡ����
    tri(i,j)=EP{1,i}(j).Cost(1);
    end
end


k=max(err,[],2);
for i=1:10
    slei=find(err(i,:)==k(i));
    Lin=leng(i,slei); %�ҵ���߷�����ȷ�ʵĳ���
    Tr=tri(i,slei); %��߷�����ȷ��ѵ��������
    trainR=min(Tr); %���ѵ����ȷ�˰�
    slT=find(Tr==trainR); %���ѵ����ȷ��λ��
    reallength=Lin(1,slT); %���ѵ���Ͳ���λ�õĳ���
    [n(i) d]=min(reallength); %��С���Ⱥ�λ��
    %[n(i) d ]=min(leng(i,find(err(i,:)==k(i))));
    nix(i)=slei(1,slT(d));
    trx(i)=Tr(1,slT(d));
    
end
n=n';
nix=nix';
trx=1-trx';
kang=[n k trx nix]; %#f test train num
sum(kang(:,1:3))