function b= DominateC2(pop1,pop2)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
 nPop=numel(pop2);
 b=true;
 for i=1:nPop
     if Dominates(pop2(i),pop1)
         b=false;
         break;
     end
 end

