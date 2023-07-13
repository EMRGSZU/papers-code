function [] = logger(message)
    disp([datestr(now), ' - ', message]);%datestr(now),展现当前的时间，disp函数是print的意思
end
