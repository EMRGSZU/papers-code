function [erCls,rClassification]=GH_accuracy...
    (S_Class,TestLables,targets,balanced)
targets=unique(TestLables);
erCls=zeros(1,numel(targets));
Erlabel=S_Class==TestLables;
for i=1:numel(targets)
    Classnum=numel(find(TestLables==targets(1,i)));
    erCls(1,i)=(Classnum-sum(Erlabel(1,TestLables==targets(1,i))))/Classnum*100;
end
if balanced
    rClassification=sum(100-erCls)/numel(targets);
else 
    rClassification=sum(Erlabel)/numel(Erlabel)*100;
end





