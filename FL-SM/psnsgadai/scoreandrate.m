function [score,rate]=scoreandrate(featurescore,position)
    score=sum(position.*featurescore);
    rate=sum(position)/size(position,2);
end