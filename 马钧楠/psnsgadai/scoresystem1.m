function [featurescore]=scoresystem1(featurelib,featureacc,n_featurelib,elite,n_feature,featurescore)
       
    featurelib1=featurelib/elite;
    featureacc=mapminmax(featureacc,0,1);
    featurelib2=featurelib1.*featureacc;
    [featurelibs,index]=sort(featurelib2,'descend');
    featurelibs=featurelibs(1:n_featurelib);
    index=index(1:n_featurelib);
    normalfeaturelibs = mapminmax(featurelibs, 0, 1);
    for k=1:n_feature
        if ismember(k,index)
            [m,n]=find(index==k);
            featurescore(k)=normalfeaturelibs(n);
        end
    end
    featurescore=featurescore*(-1);
end