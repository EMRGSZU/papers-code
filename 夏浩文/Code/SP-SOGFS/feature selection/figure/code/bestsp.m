clear all
clc

data_list=dir(fullfile(".\mytest\SPSOGFS\test\output\",'*.mat'));
number = length(data_list);
namelist = {data_list.name};
load(data_list(1).name)
bestacc = accscore;
for i =2:number
    load(strcat('.\mytest\SPSOGFS\test\output\',namelist{i}),'accscore')
    for j =1:length(accscore(i).output)
        if bestacc(i).output(j) < accscore(i).output(j)
            bestacc(i).output(j) = accscore(i).output(j);
        end
    end
end
path = strcat(".\mytest\SPSOGFS\test\output\",'bestacc.mat');
save(path,'bestacc')
        