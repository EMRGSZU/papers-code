function fitvalue = ftesterror(x,traind,cut)
% x=x(i,:);
%length=size(find(x),2);
[nl,D]=size(x);
 fitvalue=zeros(nl,3);
for i=1:nl 
tta=traind;  %�����е���ɢ��ѵ������
nindex=[];
        for k=1:size(tta,2)-1   %����ÿ������ ~ά ��k�� ת������
            %  j�����ӣ��ĵ�kά��  
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

