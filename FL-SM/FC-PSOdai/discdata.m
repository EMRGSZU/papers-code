function tmpdata = discdata(data,pop )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
tmpdata=data;
    for k=1:D   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度 
            if pop.Position(k)~=0
                for z=1:pop.Position(k)+1
                   if z==1
                       tmpdata(data(:,k)<=cut(k,z+1),k)=z;
                   elseif z==pop.Position(k)+1
                       tmpdata(data(:,k)>cut(k,z),k)=z;
                   else
                       tmpdata(data(:,k)>cut(k,z) & data(:,k)<=cut(k,z+1),k)=z;
        
                   end
                end
       
            else
                nindex=[nindex k];
            end
    end
    tmpdata(:,nindex)=[];
end

