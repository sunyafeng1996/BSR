clc;clear;
modes=["basic","blg","dlw","iho",...
    "dlw-blg","dlw-iho","iho-blg","omz8"];
num_class=10;
for i=1:length(modes)
    % 路径
    ppth="ablation_points/bsr-r34r18c10-"+modes(i)+"_points.csv";
    tpth="ablation_points/bsr-r34r18c10-"+modes(i)+"_targets.csv";
    % 加载数据
    points=importdata(ppth);
    targets=importdata(tpth);
    % 删除不合规范的行列
    points(1,:)=[];points(:,1)=[];
    targets(1,:)=[];targets(:,1)=[];
    %% 绘图
    % 样本分布
    plot_points(modes(i),points,targets,num_class)
    % 获取标签分布
    num_labels(i,:)=get_labels(targets,num_class);
end
figure(Position=[100,100,1000,250])
bar(1:length(modes),num_labels,1)
ylim([6800,15200])
l=gca;
l.YAxis .Exponent =4;
x_label={'Basic','BLG','DLW','IHO','BLG+DLW','IHO+DLW','IHO+BLG','BSR'};
set(gca, 'FontName', 'Times New Roman','FontSize',12);
set(gca,'XTickLabel',x_label,'FontSize',12,'FontName','Times New Roman','FontWeight','bold');
ylabel(sprintf('Number of samples \n in each category'),'FontSize',12,'FontName','Times New Roman','FontWeight','bold')
grid
text(0.6,14500,'× 10^4','FontSize',12,'FontName','Times New Roman','FontWeight','bold')
hold off
exportgraphics(gca,['figs/','distribution_label.pdf'],'BackgroundColor','none','ContentType','vector')

%% plot
function plot_points(mode,points,lables,num_class)
    figure(Position=[100,100,400,250])
    for i=0:1:num_class-1
        temp=points(lables==i,:);
        plot(temp(:,1), temp(:,2),'o','MarkerSize',1.5)
        hold on
    end
    xlim([min(points(:,1))-5, max(points(:,1))+5])
    ylim([min(points(:,2))-5, max(points(:,2))+5])

    set(gca, 'FontName', 'Times New Roman');  
    set(gca, 'XTickLabel', []); % 移除x轴标签  
    set(gca, 'YTickLabel', []); % 移除y轴标签 
    grid
    hold off
    exportgraphics(gca,['figs/',char(mode),'.png'],'BackgroundColor','none','ContentType','vector')
end

function dis_label=get_labels(lables,num_class)
    for i=0:1:num_class-1
        dis_label(i+1)=sum(lables==i);
    end
end
