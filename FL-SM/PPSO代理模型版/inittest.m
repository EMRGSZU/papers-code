function [cut,pindex] = inittest(data)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   data N*2 ��һ��Ϊ����ֵ���ڶ���Ϊ��ǩ��
l=size(data,1);
m=size(data,2);
data=sortrows(data,1); %������ֵ˳������

temp=data(1:end-1,2);  %��ǩֵ
temp1=data(2:end,2);   %���һ�еı�ǩֵ
%temp1(end+1,2)=temp(end,2); %�����һ�б�Ϊ��ǩֽ�����һ��

notqu=temp-temp1; %�ҵ���ǩֵ����ȵ�λ�á�
cutindex=find(notqu~=0);
pindex=cutindex;  %�и������
cut=data(cutindex,1);  %�и��






