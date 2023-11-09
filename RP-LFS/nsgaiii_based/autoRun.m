clear;clc;

DataPath = 'data\';
TrainPath='Result\';

File = dir(fullfile(DataPath,'*.mat'));
FileNames = {File.name}';
%FileNames={'wine.mat'};%测试单个数据集

t = 1;
numFiles = size(FileNames,1);
result.name=[];
result.mean=[];
result.std=[];
result.max=[];
result.min=[];
result.time=[];
resultSet=repmat(result,numFiles,t);
parfor j = 1 : t
    for i = 1 : numFiles
        FileName = erase(FileNames(i),'.mat');
        [acc, time]=main([DataPath,FileName{1},'.mat'],[TrainPath,FileName{1},'-a','-',num2str(j)]);
        resultSet(i,j).name = FileName;
        resultSet(i,j).min = min(acc);
        resultSet(i,j).max = max(acc);
        resultSet(i,j).mean = mean(acc,2);
        resultSet(i,j).std = std(acc,1);
        resultSet(i,j).time = mean(time,2);
    end
end

save([TrainPath,'result-a20.mat'],'resultSet');