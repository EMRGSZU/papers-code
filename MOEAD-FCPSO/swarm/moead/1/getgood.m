clear
load('EPI.mat')
load('matlab1.mat')
load('matlab2.mat')


for i=1:10
    for j=1:size(EP{1,i},1)
% [x(i,:),EP{i}]=PotentialPso(150,traind,cut); %返回特征选择结果
    tri(i,j)=EP{1,i}(j).Cost(1);
    end
end


k=max(err,[],2);
for i=1:10
    slei=find(err(i,:)==k(i));
    Lin=leng(i,slei); %找到最高分类正确率的长度
    Tr=tri(i,slei); %最高分类正确率训练集长度
    trainR=min(Tr); %最高训练正确了吧
    slT=find(Tr==trainR); %最高训练正确率位置
    reallength=Lin(1,slT); %最高训练和测试位置的长度
    [n(i) d]=min(reallength); %最小长度和位置
    %[n(i) d ]=min(leng(i,find(err(i,:)==k(i))));
    nix(i)=slei(1,slT(d));
    trx(i)=Tr(1,slT(d));
    
end
n=n';
nix=nix';
trx=1-trx';
kang=[n k trx nix]; %#f test train num
sum(kang(:,1:3))