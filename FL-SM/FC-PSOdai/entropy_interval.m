function entS=entropy_interval(data,cut,l_index,r_index)
% ���ܣ���ȡ������ɢ�ָ�ϵ�֮�����
% ���룺data���� n*2 ���󣬵�һ�д���ɢ����������,�ڶ��о�������
%       cut��������������ʼ�ĺ�ѡ��ɢ�ָ�ϵ㣨��ɢ���������ϵ�,�������Ҷϵ㣩
%       l_index��������ɢ�ָ�ϵ���cut�е�������0 �����������ϵ���������
%       r_index��������ɢ�ָ�ϵ���cut�е�������size(cut,2)+1 ����������Ҷϵ��Ҳ������
% �����entS����������ɢ�ָ�ϵ�֮�����

entS=0;

lrdata=getlrdata(data,cut,l_index,r_index);
n_lrdata=size(lrdata,1);
[num,value]=attrvalue(lrdata(:,2));    
for j=1:num
    p=size(find(lrdata(:,2)==value(j)),1)/n_lrdata;
    entS=entS-p*log2(p);
end    

