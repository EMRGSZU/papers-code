function cut = disc_MDLP(data)
% 功能：基于MDLP进行连续属性离散化
% 输入：data——数据矩阵，行代表实例，列代表属性，最后一列为类标签
% 输出：cut——切点
global disc;

l=size(data,1);
m=size(data,2);  
for i=1:m-1  %属性列数
        data_i=[data(:,i) data(:,m)]; 
        initcut_i=initcut(data_i);   %候选切割点，列向量
        l_index=0;
        r_index=size(initcut_i,2)+1;  
        disc=[];
        bincut_MDLP(data_i,initcut_i,l_index,r_index);%递归的分割来求切点
        disc=sort(disc);
        len=size(disc,2);
        cut(i,1)=len;
        cut(i,2:len+1)=disc;  %生成切点表

end