function cut=initcut(data)
% ���ܣ���ȡ��ʼ�ĺ�ѡ��ɢ�ָ�ϵ�
% ���룺data���� n*2 ���󣬵�һ�д���ɢ����������,�ڶ���Ϊ���ǩ
% �����cut��������������ʼ�ĺ�ѡ��ɢ�ָ�ϵ㣨��ɢ���������ϵ�,�������Ҷϵ㣩
initcutp=[];
tmp=sortrows(data,1); %������ֵ��������
a=tmp(1:end-1,end); %�������д�
b=tmp(2:end,end);
cindex=find(a~=b);  %�ҵ�����֮��ֵ��ͬ�ĵ㣬ȡ���ǵ�ƽ��ֵ��Ϊ�е�
length=size(cindex,1);
for i=1:length
    colu=cindex(i);
    initcutp=[initcutp (tmp(colu,1)+tmp(colu+1,1))/2];
end
cut=initcutp;


