clc;
clear;
mzx=[2	3	4	5	6	7	8	9	10];
mzy=[94.56	94.61	94.6	94.71	94.71	94.74	94.8	94.8	94.7];
st1x=[0.1	0.2	0.3	0.4	0.5	0.6	0.7	0.8	0.9	1];
st1y=[94.8	94.35	94.43	94.52	94.41	94.28	94.61	94.36	94.51	94.41];
st2x=[0.05	0.06	0.07	0.08	0.09	0.1	0.11	0.12	0.13	0.14	0.15];
st2y=[94.29	94.35	94.49	94.51	94.66	94.8	94.8	94.78	94.66	94.66	94.61];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(Position=[100,100,450,250]) 
bar(mzx,mzy,'FaceColor','#4682B4')
ylim([94.5,94.85])
hold on
h = findobj(gca,'Type','bar');  
plot(mzx, mzy, 'o-', 'LineWidth', 3);
set(gca, 'FontName', 'Times New Roman','FontSize',18,"FontWeight","bold");
xlabel('mz', 'FontSize', 20, 'FontName', 'Times New Roman','FontWeight','bold');  
ylabel('Accuracy', 'FontSize', 20, 'FontName', 'Times New Roman','FontWeight','bold');   
grid on;   
hold off;
exportgraphics(gca,'figs/mz.pdf','BackgroundColor','none','ContentType','vector')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(Position=[100,100,450,250]) 
bar(st1x,st1y,'FaceColor','#4682B4'); % 绘制柱状图
ylim([94.25,94.85])
hold on
h = findobj(gca,'Type','bar');  
plot(st1x, st1y, 'o-', 'LineWidth', 3);
set(gca, 'FontName', 'Times New Roman','FontSize',18,"FontWeight","bold");
xlabel('step_\alpha', 'FontSize', 20, 'FontName', 'Times New Roman','FontWeight','bold');  
ylabel('Accuracy', 'FontSize', 20, 'FontName', 'Times New Roman','FontWeight','bold');  
grid on;   
hold off;
exportgraphics(gca,'figs/step_large.pdf','BackgroundColor','none','ContentType','vector')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(Position=[100,100,450,250])  
bar(st2x,st2y,'FaceColor','#4682B4'); % 绘制柱状图
ylim([94.25,94.85])
hold on
h = findobj(gca,'Type','bar');  
plot(st2x, st2y, 'o-', 'LineWidth', 3);
set(gca, 'FontName', 'Times New Roman','FontSize',18,"FontWeight","bold");   
xlabel('step_\alpha', 'FontSize', 20, 'FontName', 'Times New Roman','FontWeight','bold');  
ylabel('Accuracy', 'FontSize', 20, 'FontName', 'Times New Roman','FontWeight','bold');  
grid on;   
hold off;
exportgraphics(gca,'figs/step_small.pdf','BackgroundColor','none','ContentType','vector')
