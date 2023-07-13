function [negFea,strFea,medFea] = isLinear(fea,range,mode)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
% fea  : n * d data matrix including n samples and d features
% range: define the range of being linear like very strongly, strongly,
%        moderately, weakly and negligibly
% mode : "z" is the zscore standardization
inFea = fea;
d = size(fea,2);

if mode == "z"
    inFea = zscore(fea);
end

covMat = cov(inFea);

rangeL = length(range);

negCov = abs(covMat) < range(1);
strCov = abs(covMat) > range(2);

negFea = []; strFea =[]; medFea = [];



num = d/4;
i = 0;
isN = 0;
toGet = (1:d)';

while length(negFea)<num && (d-length(negFea)-length(strFea))>0 
%     && length(strFea) < num
    i = i+1;
    numNeg = sum(negCov);
    numStr = sum(strCov);
    ratio = numNeg .* numStr;
    [~,index] = sort(ratio,"descend");
    strCov(index(1),index(1)) = 0;

    indStr = find(strCov(:,index(1)));
    negCov(index(1),:) = 0;
    negCov(:,index(1)) = 0;
    negCov(indStr,:) = 0;
    negCov(:,indStr) = 0;
    strCov(indStr,:) = 0;
    strCov(:,indStr) = 0;
    strCov(index(1),:) = 0;
    strCov(:,index(1)) = 0;

    negFea = [negFea;index(1)];
    strFea = [strFea;indStr];

end


medFea1 = [negFea;strFea];
medFea = find(1-ismember(toGet,medFea1));

% check 
check = [negFea;medFea;strFea];
cond1 = length(unique(check));
cond2 = length(check);
if cond1 ~= d || cond2~=d 
    disp(strcat("检查程序",num2str(cond1),num2str(cond2)))
end

end

