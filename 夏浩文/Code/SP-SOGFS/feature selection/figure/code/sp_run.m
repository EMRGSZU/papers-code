data_list=dir(fullfile(".\SP-SOGFS数据\std",'*.mat'));
number = {data_list.name};

[s,~] = listdlg('Name','Dataset','Promptstring','Select which dataset',...
    'SelectionMode','multiple','ListSize',[300,400],'liststring',number);

accscore(7).dataset = 0;
accscore(7).output = 0;

% for tryb = 1:5
    if ~isempty(s)
        for i =1:length(s)
            if ismember(number{s(i)},['BA.mat','USPSdata_20_uni.mat'])
                newfea = 1;
            else
                newfea = 0;
            end
            load(strcat(".\SP-SOGFS数据\std\",number{s(i)}),'data','label')
            disp(number{s(i)})
            
            
            accscore(i).output = sogfs_sp(data,label,newfea);
            accscore(i).dataset = number{s(i)};
            
        end
    end
    
%     path = strcat(".\mytest\SPSOGFS\test\output\",'accscore',num2str(tryb),'.mat');
%     save(path,'accscore')
% end
