function entST=entropyT(data,cut,point)
% ���ܣ���ȡ�ָ���Entropy
% ���룺data���� n*2 ���󣬵�һ�д���ɢ����������,�ڶ��б�ǩ
%       cut�����ָ��������
%       point�����ָ������
% �����entS����������ɢ�ָ�ϵ�֮�����

entS=0;

lrdata=getlrdata(data,cut,l_index,r_index);
n_lrdata=size(lrdata,1);
[num,value]=attrvalue(lrdata(:,2));    
for j=1:num
    p=size(find(lrdata(:,2)==value(j)),1)/n_lrdata;
    entS=entS-p*log2(p);
end    

