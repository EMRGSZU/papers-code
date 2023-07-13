clear all
clc

data_list=dir(fullfile(".\SP-SOGFS数据\std",'*.mat'));
number = {data_list.name};

[s,~] = listdlg('Name','Dataset','Promptstring','Select which dataset',...
    'SelectionMode','multiple','ListSize',[300,400],'liststring',number);

t_start = clock;
if ~isempty(s)
%     for tryb = 1:5
        for i =1:length(s)
            if ismember(number{s(i)},['BA.mat','USPSdata_20_uni.mat'])
                newfea = 1;
            else
                newfea = 0;
            end
            load(strcat(".\SP-SOGFS数据\std\",number{s(i)}),'data','label')
            disp(number{s(i)})
            
            
            spgamma(data,label,newfea,number{s(i)});
            
        end
end
t_end = clock;
disp(['exe time: ',num2str(etime(t_end,t_start))]);
