function tmpdata = discdata(data,pop )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
tmpdata=data;
    for k=1:D   %����ÿ������ ~ά ��k�� ת������
            %  j�����ӣ��ĵ�kά�� 
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

