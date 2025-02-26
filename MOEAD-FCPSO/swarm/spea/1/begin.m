function begin(data)
data=[data(:,2:end) data(:,1)];
datalabel=data(:,end);
total_time1 = clock;
warning off
for k=1:10
 disp(k)
EP=[];
PB=[];
% 
% EP3=[];
% PB3=[];
% EP4=[];
% PB4=[];
% EP5=[];
% PB5=[];
% EP6=[];
% PB6=[];

%??????????10??cv
Indices =crossvalind('Kfold',datalabel, 10);

cut=cell(1,10);
parfor i=1:10
test=(Indices==i);
train=~test;
traind=data(train,:);
cut{i}=disc_MDLP(traind);
end

for i=1:10
test=(Indices==i);
train=~test;
traind=data(train,:);
testd=data(test,:);
%cut=disc_MDLP(traind);
%[EP{i},PB{i},EP3{i},PB3{i},EP4{i},PB4{i},EP5{i},PB5{i},EP6{i},PB6{i}]=PotentialPso(150,traind,cut{1,i});
[EP{i},PB{i}]=PotentialPso(150,traind,cut{1,i}); %?????????????
% [length(i,:),error(i,:)] = testerror(x(i,:),traind,testd,cut) %???????????????????????
%  for j=1:size(EP{1,i},1)
% % [x(i,:),EP{i}]=PotentialPso(150,traind,cut); %?????????????
%     s=EP{1,i}(j).Position();
%    [leng(i,j),acc(i,j)] = testerror(s,traind,testd,cut{1,i}); %???????????????????????
%    
%  end
   %error(i,:)=max(acc(i,:));
   %length(i,:)=min(leng(i,find(acc(i,:)==error(i))));
end
%sum(length)/10
%sum(error)/10

%lens=strcat('lens',num2str(k),'.mat');
%accuracy=strcat('acc',num2str(k),'.mat');
indi=strcat('indi',num2str(k),'.mat');
EPI=strcat('EP',num2str(k),'.mat');
% EPI3=strcat('EP3',num2str(k),'.mat');
% EPI4=strcat('EP4',num2str(k),'.mat');
% EPI5=strcat('EP5',num2str(k),'.mat');
% EPI6=strcat('EP6',num2str(k),'.mat');
cuti=strcat('cut',num2str(k),'.mat');
pbi=strcat('pb',num2str(k),'.mat');
% pbi3=strcat('pb3',num2str(k),'.mat');
% pbi4=strcat('pb4',num2str(k),'.mat');
% pbi5=strcat('pb5',num2str(k),'.mat');
% pbi6=strcat('pb6',num2str(k),'.mat');


save(['result/',cuti], 'cut');
%save(lens,'leng');
%save(accuracy,'acc');
save(['result/',indi], 'Indices');
save(['result/',EPI],'EP');
% save(EPI3,'EP3');
% save(EPI4,'EP4');
% save(EPI5,'EP5');
% save(EPI6,'EP6');
save(['result/',pbi],'PB');
% save(pbi3,'PB3');
% save(pbi4,'PB4');
% save(pbi5,'PB5');
% save(pbi6,'PB6'); %??12??
end
total_time2 = clock;
total_time = etime(total_time2, total_time1)



