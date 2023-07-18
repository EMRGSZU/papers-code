
%基于最小描述长度的属性离散化
fid=fopen('../dataset/dataset.ini');
while feof(fid)~=1
    datasetname= fgetl(fid);                       %读数据集，每行一个    
    eval(['cd ../dataset;' datasetname ';cd ../disc_MDLP']);%加载所使用的数据文件。输出为data数据集。最后一列为决策
    eval(['dataset=disc_MDLP(dataset,feat);']);%进行连续属性离散化  
    eval(['cd ../dataset;']);
    eval(['!mkdir ' datasetname ';']);
    eval(['cd ../disc_MDLP;']);
    eval(['save ../dataset/' datasetname '/discMDLP dataset']);
end
