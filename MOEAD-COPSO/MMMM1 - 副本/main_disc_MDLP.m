
%������С�������ȵ�������ɢ��
fid=fopen('../dataset/dataset.ini');
while feof(fid)~=1
    datasetname= fgetl(fid);                       %�����ݼ���ÿ��һ��    
    eval(['cd ../dataset;' datasetname ';cd ../disc_MDLP']);%������ʹ�õ������ļ������Ϊdata���ݼ������һ��Ϊ����
    eval(['dataset=disc_MDLP(dataset,feat);']);%��������������ɢ��  
    eval(['cd ../dataset;']);
    eval(['!mkdir ' datasetname ';']);
    eval(['cd ../disc_MDLP;']);
    eval(['save ../dataset/' datasetname '/discMDLP dataset']);
end
