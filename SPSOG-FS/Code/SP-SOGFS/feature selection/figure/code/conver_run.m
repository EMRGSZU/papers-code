clear all
clc

data_list=dir(fullfile(".\SP-SOGFS数据\std",'*.mat'));
number = {data_list.name};

[s,~] = listdlg('Name','Dataset','Promptstring','Select which dataset',...
    'SelectionMode','multiple','ListSize',[300,400],'liststring',number);

dirpath = "F:\Users\cnnyl\Documents\MATLAB\mytest\SPSOGFS\test\convergence_output\";
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
            nClass = length(unique(label));
            
            [~,output] = SOGFS_sp(data,1,nClass,66);
            path = strcat(dirpath,number{s(i)}(1:end-4),'_conver.mat');
            save(path,"output")
            
        end
end
t_end = clock;
disp(['exe time: ',num2str(etime(t_end,t_start))]);
