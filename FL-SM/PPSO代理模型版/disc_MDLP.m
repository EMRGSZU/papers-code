function cut = disc_MDLP(data)
% 功能：基于MDLP进行连续属性离散化
% 输入：data——数据矩阵，行代表实例，列代表属性，最后一列为决策属性
%       feat——行向量，代表属性的类型，0代表连续属性，1代表名义型属性
% 输出：discdata——离散化后的数据集

% if size(data,2)~=size(feat,2)
%     return;
% end
% data=[data(:,2:end) data(:,1)];
global disc;

l=size(data,1);
m=size(data,2);  

%[cut,pindex] = inittest(data);




for i=1:m-1  %属性列数
   % if feat(i)==0
        data_i=[data(:,i) data(:,m)];
    
        initcut_i=initcut(data_i);   %候选切割点，列向量
               
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