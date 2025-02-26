clear;clc;

DataPath = '';
TrainPath='';

Alpha.max=200;
Alpha.min=1;
balanced=0;

FileNames={'DLBCL.mat'};
t = 1;
numFiles = size(FileNames,1);
result.name=[];
result.mean=[];
result.std=[];
result.max=[];
result.min=[];
resultSet=repmat(result,numFiles,t);

for j = 1 : t
    for i = 1 : numFiles
        FileName = erase(FileNames(i),'.mat');
        temp=demo([DataPath,FileName{1},'.mat'],[TrainPath,FileName{1},'-',num2str(j)],Alpha,balanced);
        resultSet(i,j).name = FileName;
        resultSet(i,j).min = min(temp);
        resultSet(i,j).max = max(temp);
        resultSet(i,j).mean = mean(temp,2);
        resultSet(i,j).std = std(temp,1);
    end
end

save([TrainPath,'result-a20.mat'],'resultSet');