function position= initparticle(cut,D)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    position=zeros(1,D);
    for j=1:D
        if cut(j,1)~=0     
        position(:,j)=randi([0 cut(j,1)]);
        end
    end
end

