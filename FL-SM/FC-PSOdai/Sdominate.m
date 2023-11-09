function result= Sdominate(a,b)
%UNTITLED 求非支配解
%   
    result=all(a<=b) && any(a<b);
end

