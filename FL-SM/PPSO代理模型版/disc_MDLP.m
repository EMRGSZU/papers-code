function cut = disc_MDLP(data)
% ���ܣ�����MDLP��������������ɢ��
% ���룺data�������ݾ����д���ʵ�����д������ԣ����һ��Ϊ��������
%       feat�������������������Ե����ͣ�0�����������ԣ�1��������������
% �����discdata������ɢ��������ݼ�

% if size(data,2)~=size(feat,2)
%     return;
% end
% data=[data(:,2:end) data(:,1)];
global disc;

l=size(data,1);
m=size(data,2);  

%[cut,pindex] = inittest(data);




for i=1:m-1  %��������
   % if feat(i)==0
        data_i=[data(:,i) data(:,m)];
    
        initcut_i=initcut(data_i);   %��ѡ�и�㣬������
               
%         cutsize=size(initcut_i,2);
%         for i=1:cutsize
%             
%         end
        l_index=0;
        r_index=size(initcut_i,2)+1;  
    
        disc=[];
        bincut_MDLP(data_i,initcut_i,l_index,r_index);
        disc=sort(disc);
%         disp(disc);
        len=size(disc,2);
        cut(i,1)=len;
        cut(i,2:len+1)=disc;
%         cut(i,2:end)=disc;

%         if size(disc,2)==0  
%             discdata(:,i)=1;
%         else
%             for j=1:(size(disc,2)+1)
%                 if j==1
%                     discdata(find(data(:,i)<disc(1)),i)=1;
%                 elseif j==(size(disc,2)+1)
%                     discdata(find(data(:,i)>=disc(size(disc,2))),i)=j;
%                 else
%                     discdata(find(data(:,i)>=disc(j-1) & data(:,i)<disc(j)),i)=j;
%                 end
%             end
%         end        
   % end    
end