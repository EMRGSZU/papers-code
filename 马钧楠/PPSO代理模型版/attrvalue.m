function [classnum,classlabel]=attrvalue(data)
% 功能：求取数据集每个属性的取值及个数
% 输入：data——数据集，行为实例，列为属性。
% 输出：Mnum——列向量，代表每个属性的取值个数
%       Mvalue——每一行代表一个属性的所有取值，末尾用0补齐
% 
% l=size(data,1);%行（实例）
% m=size(data,2);%列（属性）
% 
% Mnum=[];
% Mvalue=[];
% for i=1:m
% 	tmpC=data(:,i);  
% 	tmpN=0;
% 	while size(tmpC,1)~=0;  %行数
%        	Mvalue(i,(tmpN+1))=tmpC(1);
%        	tmpC=tmpC(find(tmpC~=tmpC(1)));
%        	tmpN=tmpN+1;
% 	end
% 	Mnum=[Mnum;tmpN];
% end    

classlabel=unique(data(:,end));
classnum=size(classlabel,1);
		