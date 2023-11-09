function fitvalue = ftesterror(x,traind,cut)
% x=x(i,:);
%length=size(find(x),2);
[nl,D]=size(x);
 fitvalue=zeros(nl,3);
for i=1:nl 
tta=traind;  %根据切点离散化训练集；
nindex=[];
        for k=1:size(tta,2)-1   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if x(i,k)~=0
                for z=1:x(i,k)+1
                   if z==1
                       tta(traind(:,k)<=cut(k,z+1),k)=z;
                   elseif z==x(i,k)+1
                       tta(traind(:,k)>cut(k,z),k)=z;
                   else
                       tta(traind(:,k)>cut(k,z) & traind(:,k)<=cut(k,z+1),k)=z;
                   end
                end         
            else
                nindex=[nindex k];
            end          
        end
        tta(:,nindex)=[];
     
        fitvalue(i,:)=fitness(tta,D);
end



end

