function ecvten(data,Indices)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
data=[data(:,2:end) data(:,1)];
test=(Indices==8);
train=~test;
traind=data(train,:);
testd=data(test,:);
save traindata traind;
save testdata testd;

end
% ecvten(data,Indices)
