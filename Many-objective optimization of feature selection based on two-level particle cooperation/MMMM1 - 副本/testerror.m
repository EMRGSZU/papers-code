function [length,error] = testerror(x,traind,testd,cut)
% x=x(i,:);
%testd = dicdata(testd,cut);
dindex=find(x);
length=size(dindex,2);
trainl=traind(:,end);
testl=testd(:,end);

traind=traind(:,dindex);
testd=testd(:,dindex);


Mdl=fitcknn(traind,trainl);
pclass=predict(Mdl,testd);
result=(pclass==testl);
error=size(find(result),1)/size(result,1); %查看最后的分类正确率error应为accuarcy
end

