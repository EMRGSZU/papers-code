function b= DominateC2(pop1,pop2)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
 nPop=numel(pop2);
 b=true;
 for i=1:nPop
     if Dominates(pop2(i),pop1)
         b=false;
         break;
     end
 end

