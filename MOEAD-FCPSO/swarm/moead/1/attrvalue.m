function [classnum,classlabel]=attrvalue(data)
% ���ܣ���ȡ���ݼ�ÿ�����Ե�ȡֵ������
% ���룺data�������ݼ�����Ϊʵ������Ϊ���ԡ�
% �����classnum�������ǩ������
%       classlabel�������ǩ
% 


classlabel=unique(data(:,end));
classnum=size(classlabel,1);
		