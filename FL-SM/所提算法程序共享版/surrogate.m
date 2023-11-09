function ip=surrogate(featurescore,position,mscore,mrate,srate)
    
    
    sscore=sum(position.*featurescore);
%     srate=sum(position)/size(position,2);
    
    %if mrate>0.01
    %    xrate=mrate-0.01;
    %else
    %    xrate=mrate-0.001;
    %end
    
    
    if sscore>mscore && srate<mrate
        ip=1;
        %mscore=sscore;
        %mrate=srate;
    %elseif sscore<mscore & srate<=xrate
    %        ip=1;
    else
        
            ip=0;
        
    end
            
end