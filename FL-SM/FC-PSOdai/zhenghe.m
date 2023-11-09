load('EPI.mat');
a=[];
for j=1:10
    for i=1:size(EP{1,j},1)
    a=[a ;EP{1,j}(i).Cost];
    end
end
MOEA=a;
plot3(MOEA(:,3),MOEA(:,2),MOEA(:,1),'o','Color','c','MarkerSize',5);
grid on;

