close all
clc
clear all

%% Constant
pi2=2*pi;
MHz=1000000;   % MHz to Hz conversion
msec=0.001;    % msec to sec conversion
usec=0.000001; % usec to sec conversion

F_sampling = 399996327;   % Sampling frequency
two_channel=1;          % 1 if two channels presented in data file
c = physconst('LightSpeed');

%% Load Data

% [filenaam,DataDir]=uigetfile('*.bin');
% filenaam=[DataDir,filenaam];
filenaam = '/Volumes/Personal/Backup/PMCWPARSAXData/sync_clock_data_a13/VV/VV_20191126093852.bin';   % 0.5ms data
% filenaam = '/Volumes/Personal/Backup/PMCWPARSAXData/sync_clock_data_a13/VV/VV_20191126093725.bin';   % 1ms data

% read file

fid=fopen(filenaam);
info=fread(fid, 2,'int32');
fseek(fid, 0, 'bof'); % ???

N_blocks_to_skip = 1;       % put a number of blocs which you want to skip from processing
N_blocks_to_process = 21;    % how many data-blocs to process

if (N_blocks_to_skip>0)
    for i=1:N_blocks_to_skip
        A=fread(fid, info(1),'int16');
    end
end

% read first block
A=fread(fid, info(1),'int16');
% remove header
A(1:4)=[];

% read following blocks and concatenate them into vector A
for i=1:(N_blocks_to_process-1)
    B=fread(fid, info(1),'int16');
    B(1:4)=[];
    A=[A(:);B(:)];
end

%close file
fclose(fid);

N_sample=max(size(A));

%% Pre-process

B=reshape(A,2,N_sample/2);
A=B(1,:);
B=B(2,:);
A=A(:);
B=B(:);
N_sample=max(size(A))

A=A(1:N_sample);
B=B(1:N_sample);

%time array for plotting
t=(1/400/1000).*linspace(0,N_sample,N_sample+1); % msec
t(end)=[];

%% 
% =====================================================================
% ploting of time seria data from two channels
% =====================================================================
figure;
% subplot(2,1,1)
plot(A)
xlabel('Fast time, msec')
title('ADC0, Tx')

% subplot(2,1,2)
figure
plot(B)
xlabel('Fast time, msec')
title('ADC1, Rx')


%% %%%%%%%%%%%%%%%%%% Comment this when process 1.0ms data %%%%%%%%%%%%%%%%
% split into fast-time and slow-time
% for 0.5ms data
starting_point = 203066; 
slowtime_length = 199997;                       
slowtime = floor(N_blocks_to_process/1) -1 ;
end_point = starting_point + slowtime_length * slowtime - 1;
received = B(starting_point:end_point);
transmitted = A(starting_point:end_point);
% clear B;
data = reshape(received, [slowtime_length, slowtime]);
tdata = reshape(transmitted, [slowtime_length, slowtime]);

tdata(:, 1:2:end) = circshift(tdata(:, 1:2:end), 1, 1);

figure()
imagesc(real(tdata(:, :)))
title('Data real part')
colorbar()
colormap(jet)

figure()
imagesc(abs(fftshift(fft(tdata(1:1024, :), 128, 2), 2)))
title('FFT on slow-time')
colorbar()
colormap(jet)

%% %%%%%%%%%%%%%%%%%% Comment this when process 0.5ms data %%%%%%%%%%%%%%%%
% % split into fast-time and slow-time
% % for 1ms data 
% starting_point = 200961; 
% slowtime_length = 399994;                       
% slowtime = floor(N_blocks_to_process/2);
% end_point = starting_point + slowtime_length * slowtime - 1;
% received = B(starting_point:end_point);
% transmitted = A(starting_point:end_point);
% % clear B;
% data = reshape(received, [slowtime_length, slowtime]);
% tdata = reshape(transmitted, [slowtime_length, slowtime]);
% 
% 
% figure()
% imagesc(real(tdata(:, :)))
% title('Data real part')
% colorbar()
% colormap(jet)
% 
% 
% figure()
% imagesc(abs(fftshift(fft(tdata(1:1024, :), 128, 2), 2)))
% title('FFT on slow-time')
% colorbar()
% colormap(jet)
