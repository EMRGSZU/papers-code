function featurescore=scoresystem2(featurelib,featureacc,elite,featurescore)
       
    featurelib1=featurelib/elite;
    featureacc=mapminmax(featureacc,0,1);
    featurelib2=featurelib1+featureacc;
    featurescore = mapminmax(featurelib2, 0, 1);
    featurescore=-1*featurescore;
end