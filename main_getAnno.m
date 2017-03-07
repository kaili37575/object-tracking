% get ground truth boxes

%strDirName='seq1';
strDirName='seq2';

D=dir([strDirName '/*.jpg']);

%be careful, you may overwrite the existing db file. 
isOverwrite=false;
    
if isOverwrite 
    gd=cell(1,length(D));
else
    load([strDirName '.mat'],'gd'); 
end
for i=1:length(D)
    disp(i);
    I=imread([strDirName '/' D(i).name]);
    figure(1), 
    imshow(I);
    hold on;
    if isOverwrite
        % generate new annotations
        fprintf('frame %d, click top-left and bottom-right corners...',i);
        [gx,gy]=ginput(2);
        %save
        gd{i}=[gx, gy];                 
    end
    % visualize
    if ~isempty(gd{i})
        gx=gd{i}(:,1);
        gy=gd{i}(:,2);
        %visual the box;
        plot([gx(1) gx(2) gx(2) gx(1) gx(1)],[gy(1) gy(1) gy(2) gy(2) gy(1)],'.-','LineWidth',5, 'Color','g');
    end    
    pause(1/20);
end
if isOverwrite
    save([strDirName '.mat'],'gd');
end