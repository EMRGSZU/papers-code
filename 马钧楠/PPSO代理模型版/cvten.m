function cvten( data)
%进行5倍cv分割
%   此处显示详细说明
data=[data(:,2:end) data(:,1)];
datalabel=data(:,end);
Indices =crossvalind('Kfold',datalabel, 10);
test=(Indices==1);
train=~test;
traind=data(train,:);
testd=data(test,:);
save cvindex Indices;
save traindata traind;
save testdata testd;


