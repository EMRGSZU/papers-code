function [classnum,classlabel]=attrvalue(data)
% ���ܣ���ȡ���ݼ�ÿ�����Ե�ȡֵ������
% ���룺data�������ݼ�����Ϊʵ������Ϊ���ԡ�
% �����Mnum����������������ÿ�����Ե�ȡֵ����
%       Mvalue����ÿһ�д���һ�����Ե�����ȡֵ��ĩβ��0����
% 
% l=size(data,1);%�У�ʵ����
% m=size(data,2);%�У����ԣ�
% 
% Mnum=[];
% Mvalue=[];
% for i=1:m
% 	tmpC=data(:,i);  
% 	tmpN=0;
% 	while size(tmpC,1)~=0;  %����
%        	Mvalue(i,(tmpN+1))=tmpC(1);
%        	tmpC=tmpC(find(tmpC~=tmpC(1)));
%        	tmpN=tmpN+1;
% 	end
% 	Mnum=[Mnum;tmpN];
% end    

classlabel=unique(data(:,end));
classnum=size(classlabel,1);
		