clc; close all; clear all;

%% Load Data
Mouse_2_class = csvread('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\Datasets\1D\Predictions\1DCNN_Mouse_2_pre_intercap_class_predictions');
Mouse_2_class = (Mouse_2_class + 1) / 3;

% Load Mask
Mask_Slice_1 = imread('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Masks\Pre\Full_Slice\Mask_Slice_1.tif');
Mask_Slice_2 = imread('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Masks\Pre\Full_Slice\Mask_Slice_2.tif');
Mask_Slice_3 = imread('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Masks\Pre\Full_Slice\Mask_Slice_3.tif');
Mask_Slice_4 = imread('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Masks\Pre\Full_Slice\Mask_Slice_4.tif');
Mask_Slice_5 = imread('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Masks\Pre\Full_Slice\Mask_Slice_5.tif');

% Load Anatomy
Mouse_2_Anatomy_Slice_1 = load('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Anatomy\Pre\slice1.mat');
Mouse_2_Anatomy_Slice_1 = Mouse_2_Anatomy_Slice_1.slice1;

Mouse_2_Anatomy_Slice_2 = load('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Anatomy\Pre\slice2.mat');
Mouse_2_Anatomy_Slice_2 = Mouse_2_Anatomy_Slice_2.slice2;

Mouse_2_Anatomy_Slice_3 = load('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Anatomy\Pre\slice3.mat');
Mouse_2_Anatomy_Slice_3 = Mouse_2_Anatomy_Slice_3.slice3;

Mouse_2_Anatomy_Slice_4 = load('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Anatomy\Pre\slice4.mat');
Mouse_2_Anatomy_Slice_4 = Mouse_2_Anatomy_Slice_4.slice4;

Mouse_2_Anatomy_Slice_5 = load('C:\Users\bjcampb\Google Drive\Datasets_ChemComp\August\Mouse 2 - Intercap\Anatomy\Pre\slice5.mat');
Mouse_2_Anatomy_Slice_5 = Mouse_2_Anatomy_Slice_5.slice5;

%% Map Score Data to Mask%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

j = 1;
mapM1Slice1 = zeros(size(Mask_Slice_1));
mapM1Slice2 = zeros(size(Mask_Slice_2));
mapM1Slice3 = zeros(size(Mask_Slice_3));
mapM1Slice4 = zeros(size(Mask_Slice_4));
mapM1Slice5 = zeros(size(Mask_Slice_5));

M1_Mask_Slice1 = Mask_Slice_1';
M1_Mask_Slice2 = Mask_Slice_2';
M1_Mask_Slice3 = Mask_Slice_3';
M1_Mask_Slice4 = Mask_Slice_4';
M1_Mask_Slice5 = Mask_Slice_5';


M1_Mask_Slice1 = double(M1_Mask_Slice1);
M1_Mask_Slice2 = double(M1_Mask_Slice2);
M1_Mask_Slice3 = double(M1_Mask_Slice3);
M1_Mask_Slice4 = double(M1_Mask_Slice4);
M1_Mask_Slice5 = double(M1_Mask_Slice5);



for i=1:(size(M1_Mask_Slice1,1)*size(M1_Mask_Slice1,2));
    if M1_Mask_Slice1(i) == 1;
%         mapM1Slice3(i) = M1P195score(j,2);
        mapM1Slice1(i) = Mouse_2_class(j);
        j = j+1;
    end;
end;

for i=1:(size(M1_Mask_Slice2,1)*size(M1_Mask_Slice2,2));
    if M1_Mask_Slice2(i) == 1;
%         mapM1Slice3(i) = M1P195score(j,2);
        mapM1Slice2(i) = Mouse_2_class(j);
        j = j+1;
    end;
end;

for i=1:(size(M1_Mask_Slice3,1)*size(M1_Mask_Slice3,2));
    if M1_Mask_Slice3(i) == 1;
%         mapM1Slice3(i) = M1P195score(j,2);
        mapM1Slice3(i) = Mouse_2_class(j);
        j = j+1;
    end;
end;


for i=1:(size(M1_Mask_Slice4,1)*size(M1_Mask_Slice4,2));
    if M1_Mask_Slice4(i) == 1;
%         mapM1Slice4(i) = M1P195score(j,2);
        mapM1Slice4(i) = Mouse_2_class(j);
        j = j+1;
    end;
end;

for i=1:(size(M1_Mask_Slice5,1)*size(M1_Mask_Slice5,2));
    if M1_Mask_Slice5(i) == 1;
%         mapM1Slice5(i) = M1P195score(j,2);
        mapM1Slice5(i) = Mouse_2_class(j);
        j = j+1;
    end;
end;


%%
M1_Mask_Slice1 = M1_Mask_Slice1';
M1_Mask_Slice2 = M1_Mask_Slice2';
M1_Mask_Slice3 = M1_Mask_Slice3';
M1_Mask_Slice4 = M1_Mask_Slice4';
M1_Mask_Slice5 = M1_Mask_Slice5';

mapM1Slice1 = mapM1Slice1';
mapM1Slice2 = mapM1Slice2';
mapM1Slice3 = mapM1Slice3';
mapM1Slice4 = mapM1Slice4';
mapM1Slice5 = mapM1Slice5';

%%
bwM1Mask1 = M1_Mask_Slice1;
bwM1Mask2 = M1_Mask_Slice2;
bwM1Mask3 = M1_Mask_Slice3;
bwM1Mask4 = M1_Mask_Slice4;
bwM1Mask5 = M1_Mask_Slice5;

AnatomyM1Slice1 = Mouse_2_Anatomy_Slice_1/(max(Mouse_2_Anatomy_Slice_1(:)));
AnatomyM1Slice2 = Mouse_2_Anatomy_Slice_2/(max(Mouse_2_Anatomy_Slice_2(:)));
AnatomyM1Slice3 = Mouse_2_Anatomy_Slice_3/(max(Mouse_2_Anatomy_Slice_3(:)));
AnatomyM1Slice4 = Mouse_2_Anatomy_Slice_4/(max(Mouse_2_Anatomy_Slice_4(:)));
AnatomyM1Slice5 = Mouse_2_Anatomy_Slice_5/(max(Mouse_2_Anatomy_Slice_5(:)));

rgbM1Slice1 = AnatomyM1Slice1(:,:,[1 1 1]);
rgbM1Slice2 = AnatomyM1Slice2(:,:,[1 1 1]);
rgbM1Slice3 = AnatomyM1Slice3(:,:,[1 1 1]);
rgbM1Slice4 = AnatomyM1Slice4(:,:,[1 1 1]);
rgbM1Slice5 = AnatomyM1Slice5(:,:,[1 1 1]);

%% Plot
figure;
subplot(2,3,1);
imshow(mapM1Slice1);
colormap(flipud(parula));
hold on;
h = imshow(rgbM1Slice1);
set(h, 'AlphaData', ~bwM1Mask1);


subplot(2,3,2);
imshow(mapM1Slice2);
colormap(flipud(parula));
hold on;
h = imshow(rgbM1Slice2);
set(h, 'AlphaData', ~bwM1Mask2);

subplot(2,3,3);
imshow(mapM1Slice3);
colormap(flipud(parula));
hold on;
h = imshow(rgbM1Slice3);
set(h, 'AlphaData', ~bwM1Mask3);

subplot(2,3,4);
imshow(mapM1Slice4);
colormap(flipud(parula));
hold on;
h = imshow(rgbM1Slice4);
set(h, 'AlphaData', ~bwM1Mask4);

subplot(2,3,5);
imshow(mapM1Slice5);
colormap(flipud(parula));
hold on;
h = imshow(rgbM1Slice5);
set(h, 'AlphaData', ~bwM1Mask5);



% colorbar;




