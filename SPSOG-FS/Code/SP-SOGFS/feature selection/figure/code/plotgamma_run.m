clear all
clc
path_da =".\mytest\SPSOGFS\test\gamma_output\";
data_list=dir(path_da);
trash_in = {data_list.name};
dir_name = trash_in(3:end);

[s,~] = listdlg('Name','Dataset','Promptstring','Select which dataset',...
    'SelectionMode','multiple','ListSize',[300,400],'liststring',dir_name);

gammaCandi = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3];
feanum={10:10:100,50:50:300};

t_start =clock;
if ~isempty(s)
%     for tryb = 1:5
        for i =1:length(s)
            z_acc =[];
            
            output = dir(fullfile(strcat(path_da,dir_name{s(i)},'\*.mat')));
            output_list ={output.name};
            for j =1:length(output_list)
                load(strcat(path_da,dir_name{s(i)},'\',output_list{j}))
                z_acc= [z_acc;mtrResult(2,:)];
            end
            test =z_acc';
            if numel(z_acc) == 70
                y_fea = feanum{1};                
            else
                y_fea = feanum{2};
            end
            ybins = y_fea;
            title('Grouped Style')
            xlabel('parameter \gamma');        
            ylabel('#features');     
            zlabel('ACC');
            set(gca,'YTickLabel',ybins);
            bar3(ybins,z_acc')
            
            saveas(gcf,strcat("F:\Users\cnnyl\Desktop\gamma\",dir_name{s(i)}(1:end-4),'.png'))
            disp(dir_name{s(i)})

            
        end
end
t_end = clock;
disp(['exe time: ',num2str(etime(t_end,t_start))]);
