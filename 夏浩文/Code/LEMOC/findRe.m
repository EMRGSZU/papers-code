function [outputArg1,outputArg2] = findRe(inputArg1)
%FINDRE 此处显示有关此函数的摘要
%   此处显示详细说明
d = max(size(inputArg1));
outputArg1=[];
outputArg2=[];
for i = 1:d
    for j = i+1:d
        if inputArg1(i)==inputArg1(j)
            outputArg1 = [outputArg1;inputArg1(i)];
            outputArg2 = [outputArg2;[i,j]];
        end
    end
end

end

