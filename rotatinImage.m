clc;    % Clear the command window.
fprintf('Beginning to run %s.m.\n', mfilename);
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
imageFolder_train = fullfile('F:\matlab\skin datasert\matlab\Training_Data\train');
imds_train = imageDatastore(fullfile(imageFolder_train), 'LabelSource', 'foldernames');
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 18;
P1 = [ 0 1
	-1 0];
P2 = [0 -1
	1  0];
% th = 45;
% A = [cosd(th)   -sind(th); sind(th) cosd(th)];
x = 45:315
for n=1:1   %379
    th = round (100*rand(1));
A = [cosd(th)   -sind(th); sind(th) cosd(th)];
    n
u_train = readimage(imds_train,n);
% figure(1)
% subplot(2,1,1)
% image(u_train);
axis equal
[m1,m2,m3]=size(u_train);
[xmax, ymax,q1,q2] = frame_adjust(m1,m2,A,P1,P2);
y = zeros(xmax,ymax,3)+255;
for i=1:m1
	fprintf('Row = %d\n', i);
	for j=1:m2
		% 		fprintf('Row = %d, column = %d\n', i, j);
		v = ceil(P2*A*P1*[i,j]'+[q1,q2]');
		if(v(1)==0); v(1)=1; end
		if(v(2)==0); v(2)=1; end
		y(v(1),v(2),:)=u_train(i,j,:);
        

		
		%this part will plot pixel by pixel
		
% 				subplot(2,1,2)
% 				image(uint8(y));
% 				axis equal
% 				drawnow;
	end
end
% 
%         FileName = sprintf('F:\\matlab\\skin datasert\\matlab\\hh\\%d.jpg', n+600);
%         imwrite(uint8(y),FileName);

end
subplot(2,1,2)
image(uint8(y));
%Im
axis equal
drawnow;
fprintf('Done running %s.m.\n', mfilename);


function [xmax,ymax,q1,q2] = frame_adjust(m1,m2,A,P1,P2)
w1 = [m1  m1  0
	0   m2 m2];
w2 = P2*A*P1*w1;
ind1 = find(w2(1,:)<0);
ind2 = find(w2(2,:)<0);
xmax = ceil(max(w2(1,:)));
ymax = ceil(max(w2(2,:)));
q1 = 0;
q2 = 0;
if length(ind1)>0
	q1 = -min(w2(1,find(w2(1,:)<0)));
	xmax = ceil(max(w2(1,:)+q1));
end
if length(ind2)>0
	q2 = -min(w2(2,find(w2(2,:)<0)));
	ymax = ceil(max(w2(2,:)+q2));
end
end