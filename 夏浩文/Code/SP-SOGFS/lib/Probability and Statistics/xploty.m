function xploty(x,y)
%XPLOTY 画出横轴为x,纵轴为y的图像
%   此处显示详细说明
plot(x,y, 'o', 'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',5)
xlabel('x','FontSize',20,'Fontname','Times newman')
ylabel('y','Rotation',0,'FontSize',20,'Fontname','Times newman')
% xlim([-inf,inf])
% ylim([0 1])
title('Diagram','FontSize',15.0)
end

