function entSA=entropyall(data)
% ���ܣ��󼯺�S��entropy
% ���룺data���� ���ݼ������һ��Ϊ���ݼ���ǩ��Ϊ������ԣ��ݲ����ú����������ݱ�ǩ������
%      Ĭ�ϱ�ǩ����Ϊ4 Ϊ0~3
% �����entSA���� �������ݼ���Entropy
entS=0;
m=size(data,1);
for i=0:3
    p=size(find(data(:,end)==i),1)/m;
    entS=entS-p*log2(p);
end
entSA=entS;

 

