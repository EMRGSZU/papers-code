function x = fessx( all,D,dindex)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ll=size(all,2);
dl=size(dindex,2);
x=zeros(dl,D);
for j=1:dl
    dsiz=size(dindex{1,j},1);
for i=1:dsiz
    tmp= tabulate(all{1,j}(:,i));  %该特征中所选个数最多的特征
    %tmp0=find(tmp(:,1)==0);
    %if ~isempty(tmp0)&tmp(tmp0,2)>ll*2/3
       % x(j,dindex(i))=0;
    %else
    %tmpin= find(tmp(:,1)~=0);
    %[~,realin]=max(tmp(tmpin,2));
    x(j,dindex{1,j}(i))=tmp(end,1);
   % end
end
end

