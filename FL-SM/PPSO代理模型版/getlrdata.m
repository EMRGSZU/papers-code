function lrdata=getlrdata(data,cut,l_index,r_index)
% ���ܣ���ȡ������ɢ�ָ�ϵ�֮�������
% ���룺data���� n*2 ���󣬵�һ�д���ɢ����������,�ڶ��о�������
% �����lrdata����������ɢ�ָ�ϵ�֮�������
             %6
if r_index>size(cut,2) %�е����
    tmpindex=1:size(data,1);  %1~��������
else
    tmpindex=find(data(:,1)<cut(r_index));   
end
tmpdata=data(tmpindex,:);  %���бȵ�һ����������
if l_index<1
    tmp=min(data(:,1));  %����������Сֵ
    lrdata=tmpdata(find(data(tmpindex,1)>=tmp(1)),:);
else
    lrdata=tmpdata(find(data(tmpindex,1)>=cut(l_index)),:);
end
