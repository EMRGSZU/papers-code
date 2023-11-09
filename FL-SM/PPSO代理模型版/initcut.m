function cut=initcut(data)
% 功能：求取初始的候选离散分割断点
% 输入：data—— n*2 矩阵，第一列待离散的条件属性,第二列决策属性
% 输出：cut——行向量，初始的候选离散分割断点（离散间隔包括左断点,不包括右断点）
initcutp=[];
tmp=sortrows(data,1);
a=tmp(1:end-1,end);
b=tmp(2:end,end);
cindex=find(a~=b);
length=size(cindex,1);
for i=1:length
    colu=cindex(i);
    initcutp=[initcutp (tmp(colu,1)+tmp(colu+1,1))/2];
end
cut=initcutp;

% l=size(data,1);
% m=size(data,2);
% 
% initcutp=[];
% initcutflag=0;
% tmp=sortrows([data(:,1) data(:,m)],1);
% tmp1=tmp(1,2);
% for j=1:l
%     if tmp(j,2)~=tmp1
%         if initcutflag==1                
%             if tmp(j,1)~=initcutp(size(initcutp,2))
%                 initcutp=[initcutp tmp(j,1)];
%             end
%         else
%             initcutp=[initcutp tmp(j,1)];
%             initcutflag=1;
%         end
%         tmp1=tmp(j,2);
%     end
% end
% cut=initcutp;
