function position= initparticle(cut,D)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    position=zeros(1,D);
    for j=1:D
        if cut(j,1)~=0     
        position(:,j)=randi([0 cut(j,1)]);
        end
    end
end

