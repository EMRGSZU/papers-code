clear
w=0.5;
u=0.5;
v=0.02;
ttttt=[w u v];
trp=dir;
%LUE=(length(trp)-2)/4;
LUE=10;

load('cut1.mat')
lajip=size(cut{1,1},1);

switch lajip
    case 2308
        load('data/SRBCT.mat')
    case 5469
        load('data/DLBCL.mat')
    case 5726
        load('data/9_Tumors.mat')
    case 5920
        load('data/Brain_Tumor1.mat')
    case 10367
        load('data/Brain_Tumor2.mat')
    case 5327
        load('data/Leukemia1.mat')
    case 11225
        load('data/Leukemia2.mat')
    case 10509
        load('data/Prostate_Tumor.mat')
    case 12600
        load('data/Lung_Cancer.mat')
    case 12533
        load('data/11_Tumors.mat')
end
%load('Lung_Cancer.mat')
%load('Brain_Tumor1.mat')
%load('Leukemia2.mat')
%load('Prostate_Tumor.mat')
%load('9_Tumors.mat')
%load('Brain_Tumor1.mat')
%load('Brain_Tumor2.mat')
%load('DLBCL.mat')
%load('Leukemia1.mat')
%load('SRBCT.mat')
%load('11_Tumors.mat')
%thre=0.02;
data=[data(:,2:end) data(:,1)];
datalabel=data(:,end);
D=size(data,2)-1;




flag=0;

if flag==0
for k=1:LUE
tri=[];
tdis=[];
tleng=[];
traina=[];
testa=[];
n=[];
EPS=[];
cutEP=strcat('EP',num2str(k),'.mat');
%cutacc=strcat('acc',num2str(k),'.mat');
cut=strcat('indi',num2str(k),'.mat');
cutreal=strcat('cut',num2str(k),'.mat');
%cutlen=strcat('lens',num2str(k),'.mat');
ppb=strcat('pb',num2str(k),'.mat');
load(cutEP)
%load(cutacc)
load(cut)
load(cutreal)
%load(cutlen)
load(ppb)

% for i=1:10
%     EPleng(i)=size(EP{1,i},1);
% end

%infindex=find(leng==0);
for i=1:10
    EPS(i)=size(PB{1,i},1);
end
for i=1:10
    for j=1:EPS(i)
% [x(i,:),EP{i}]=PotentialPso(150,traind,cut); %返回特征选择结果
    tri(i,j)=PB{1,i}(j).Cost(1);
    tdis(i,j)=PB{1,i}(j).Cost(2);
    tleng(i,j)= PB{1,i}(j).Cost(3);
    end
end

a=[];
b=[];
c=[];
cutaa=[];
for i=1:10
    allin=[];
    asi=[];
    kk=min(tri(i,:),[],2);
    thre=kk+0.01;
    %thre=kk;
    sleceindex=find(tri(i,:)<=thre); %最大训练精度下表
    tmpsk=size(sleceindex,2);
    for l=1:tmpsk
        alls(l,:)=PB{1,i}(sleceindex(l)).Position;
        allin=[allin find(alls(l,:)>0)];
    end
    test=(Indices==i);
    train=~test;
    traind=data(train,:);
    testd=data(test,:);
    ta=tabulate(allin);

    usea=ta(:,2);
    
    usea(usea==0)=-inf;
    
    douindex1=find(usea>=floor(tmpsk/2));
    douindex2=find(usea>=round(tmpsk/2)-2);%不合理的，但是有效
    douindex3=find(usea>=round(tmpsk/3)*2); 
    douindex4=find(usea>1); 
    douindex5=find(usea>=round(tmpsk/3));
    tmpd=cell(1,5);
    asi=cell(1,5);
    tmpd{1,1}=douindex1;
    tmpd{1,2}=douindex2;
    tmpd{1,3}=douindex3;
    tmpd{1,4}=douindex4;
    tmpd{1,5}=douindex5;
     for jk=1:5
        for q=1:tmpsk
        asi{1,jk}=[asi{1,jk};alls(q,tmpd{1,jk})];
    %a1=alls(1,douindex);
   % a2=alls(2,douindex);
        end
     end
    
     
%        tttkkk=7;
%     douindex1=find(usea>=floor(tmpsk/2));
%     douindex2=find(usea>=round(tmpsk/2)-2);%不合理的，但是有效
%     douindex3=find(usea>=round(tmpsk/3)*2); 
%     douindex4=find(usea>=round(tmpsk/3));
%      douindex5=find(usea>=round(tmpsk/4)*3);
%       douindex6=find(usea>=round(tmpsk/5)*4);
%       douindex7=find(usea>=round(tmpsk/6)*5);
%    % douindex6=find(usea>=round(tmpsk/4));
%     tmpd=cell(1,tttkkk);
%     asi=cell(1,tttkkk);
%     tmpd{1,1}=douindex1;
%     tmpd{1,2}=douindex2;
%     tmpd{1,3}=douindex3;
%     tmpd{1,4}=douindex4;
%     tmpd{1,5}=douindex5;
%    tmpd{1,6}=douindex6;
%    tmpd{1,7}=douindex7;
%      for jk=1:tttkkk
%         for q=1:tmpsk
%         asi{1,jk}=[asi{1,jk};alls(q,tmpd{1,jk})];
%     %a1=alls(1,douindex);
%    % a2=alls(2,douindex);
%         end
%      end
    %x=zeros(1,D);
    %x(:,douindex)=max(asi,[],1);
    x=fessx(asi,D,tmpd);
    
     fitn = ftesterror(x,traind,cut{i});
     %allf=fitn.*ttttt;
    % allf=sum(allf,2);
    % [~,tti]=min(allf);
    [ttc,tti]=min(fitn(:,1));
    tacc=ttc;
    tttacc=find(fitn(:,1)==ttc);
    if size(tttacc,1)>1
       [imt,imc]=min(fitn(tttacc,2));
       finalin=tttacc(imc);
    else
        finalin=tti;
    end
    
    %x=zeros(1,5726);
    %x(:,douindex)=asi(4,:);
    x1=x(finalin,:);
    [length,error] = testerror(x1,traind,testd,cut{i});
    %error;
   % tmpmaxt=acc(i,sleceindex);
    %max(tmpmaxt);
    %leng(i,sleceindex);
    a(i)=error;
    b(i)=length;
    c(i)=1-tacc;
    seindex=find(x1);
    
    tempf=x1(seindex);%所选特征值的切点
    ascut(i)=size(find(tempf>1),2); %所选特征中切点之大于1的值 
    cutaa(i)=size(find(cut{i}(:,1)>1),1); %一共有多少总的大于2的数值
            %所选的特征大于2的值
    seci=cut{i}(seindex,1);        %所选特征的切点值
       %所选特征 
    allcuts(i)=size(find(seci>1),1);  %选择特征中全部大于1的值。 
    
    
    %sizefind(cut{i}(:,1)>0)
   
end


%n=n';
%traina=1-traina';
%testa=testa';
%kang=[testa]; %#f test train num
wudi(k,1)=sum(a)/10;  
wudi(k,2)=sum(c)/10;
wudi(k,3)=sum(b)/10;
wudi(k,4)=sum(ascut)/10;   %选择的进行多点离散的值 
wudi(k,5)=sum(allcuts)/10;   %选择特征切点数大于1的值
wudi(k,6)=sum(cutaa)/10;  %每个特征的切点值大于1的值
%wudi(k,7)=sum(cutaa)/10;  %不采用多点离散的正确率



end

else
    
    
    
for k=1:LUE
tri=[];
tdis=[];
tleng=[];
traina=[];
testa=[];
n=[];
EPS=[];
cutEP=strcat('EP',num2str(k),'.mat');
cutacc=strcat('acc',num2str(k),'.mat');
cut=strcat('indi',num2str(k),'.mat');
cutreal=strcat('cut',num2str(k),'.mat');
cutlen=strcat('lens',num2str(k),'.mat');
load(cutEP)
load(cutacc)
load(cut)
load(cutreal)
load(cutlen)
% for i=1:10
%     EPleng(i)=size(EP{1,i},1);
% end

%infindex=find(leng==0);
for i=1:10
    EPS(i)=size(EP{1,i},1);
end
for i=1:10
    for j=1:EPS(i)
% [x(i,:),EP{i}]=PotentialPso(150,traind,cut); %返回特征选择结果
    tri(i,j)=EP{1,i}(j).Cost(1);
    tdis(i,j)=EP{1,i}(j).Cost(2);
    tleng(i,j)= EP{1,i}(j).Cost(3);
    end
end
for i=1:10
tri(i,EPS(i)+1:end)=NaN;
%tri=1-tri;
tdis(i,EPS(i)+1:end)=NaN;
tleng(i,EPS(i)+1:end)=NaN;
%err(i,EPS(i)+1:end)=NaN;
err(i,EPS(i)+1:end)=NaN;
end
kk=min(tri,[],2);
a=[];
for i=1:10
    allin=[];
    asi=[];
    sleceindex=find(tri(i,:)==kk(i)); %最大训练精度下表
    tmpsk=size(sleceindex,2);
    for l=1:tmpsk
        alls(l,:)=EP{1,i}(sleceindex(l)).Position;
        allin=[allin find(alls(l,:)>0)];
    end
    test=(Indices==i);
    train=~test;
    traind=data(train,:);
    testd=data(test,:);
    ta=tabulate(allin);
    usea=ta(:,2);
    usea(usea==0)=-inf;
    if tmpsk<=3
    douindex=find(usea>=floor(tmpsk/2));
    else
    douindex=find(usea>=floor(tmpsk/2)-2); 
    end
    for q=1:tmpsk
        asi=[asi;alls(q,douindex)];
    %a1=alls(1,douindex);
   % a2=alls(2,douindex);
    end
    %x=zeros(1,D);
    %x(:,douindex)=max(asi,[],1);
    x=fessx(asi,D,douindex);
    %x=zeros(1,5726);
    %x(:,douindex)=asi(4,:);
    
    [length,error] = testerror(x,traind,testd,cut{i});
    err(i,sleceindex);
    leng(i,sleceindex);
    a(i)=error;
    b(i)=length;
   
end


%n=n';
%traina=1-traina';
%testa=testa';
%kang=[testa]; %#f test train num
wudi(k,1)=sum(a)/10;
wudi(k,2)=sum(b)/10;
end
    
end


wudi(end+1,:)=sum(wudi)/LUE;
wudi(end+1,:)=max(wudi(1:LUE,:));
%wudi(end+1,:)=std(wudi(1:LUE,:));
wudi(end-1:end,:)

