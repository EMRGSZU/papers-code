function featurescore=scoresystem4(featurelib,featureacc,elite,featurescore)
    featureacc=featureacc./featurelib;   
    featurelib1=featurelib/elite;
    featurelib2=featurelib1.*featureacc;
    featurescore = mapminmax(featurelib2, 0, 1);
end