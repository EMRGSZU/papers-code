function cut = disc_MDLP(data)
% ���ܣ�����MDLP��������������ɢ��
% ���룺data�������ݾ����д���ʵ�����д������ԣ����һ��Ϊ���ǩ
% �����cut�����е�
global disc;

l=size(data,1);
m=size(data,2);  
for i=1:m-1  %��������
        data_i=[data(:,i) data(:,m)]; 
        initcut_i=initcut(data_i);   %��ѡ�и�㣬������
        l_index=0;
        r_index=size(initcut_i,2)+1;  
        disc=[];
        bincut_MDLP(data_i,initcut_i,l_index,r_index);%�ݹ�ķָ������е�
        disc=sort(disc);
        len=size(disc,2);
        cut(i,1)=len;
        cut(i,2:len+1)=disc;  %�����е��

end