function ecvten(data,Indices)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
data=[data(:,2:end) data(:,1)];
test=(Indices==8);
train=~test;
traind=data(train,:);
testd=data(test,:);
save traindata traind;
save testdata testd;

end
% ecvten(data,Indices)
