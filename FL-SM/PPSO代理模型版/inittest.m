function [cut,pindex] = inittest(data)
%UNTITLED 此处显示有关此函数的摘要
%   data N*2 第一列为特征值，第二列为标签。
l=size(data,1);
m=size(data,2);
data=sortrows(data,1); %按特征值顺序排列

temp=data(1:end-1,2);  %标签值
temp1=data(2:end,2);   %相差一行的标签值
%temp1(end+1,2)=temp(end,2); %将最后一行变为标签纸的最后一个

notqu=temp-temp1; %找到标签值不相等的位置。
cutindex=find(notqu~=0);
pindex=cutindex;  %切割点坐标
cut=data(cutindex,1);  %切割点






