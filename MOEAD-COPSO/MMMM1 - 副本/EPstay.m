function b = EPstay(EP,lastEP)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    b=0;
    EPs=size(EP,1);
    lastEPs=size(lastEP,1);
    for i=1:EPs
        for j=1:lastEPs
            if Dominates(EP,lastEP)
                b=1;
                return;
            end
        end
    end
end

