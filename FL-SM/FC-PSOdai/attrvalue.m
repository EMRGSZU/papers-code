function [classnum,classlabel]=attrvalue(data)
% 功能：求取数据集每个属性的取值及个数
% 输入：data——数据集，行为实例，列为属性。
% 输出：classnum——类标签的数量
%       classlabel——类标签
% 


classlabel=unique(data(:,end));
classnum=size(classlabel,1);
		