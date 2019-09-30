close all
clc
clear all

%% Constant
pi2=2*pi;
MHz=1000000;   % MHz to Hz conversion
msec=0.001;    % msec to sec conversion
usec=0.000001; % usec to sec conversion

F_sampling = 400*MHz;   % Sampling frequency
two_channel=1;          % 1 if two channels presented in data file
c = physconst('LightSpeed');

%% Load Data

% [filenaam,DataDir]=uigetfile('*.bin');
% filenaam=[DataDir,filenaam];
filenaam = '/Users/shengzhixu/Documents/ExperimentalData/pmcw/cars/HH_20190919134453.bin';

% read file

fid=fopen(filenaam);
info=fread(fid, 2,'int32');
fseek(fid, 0, 'bof'); % ???

N_blocks_to_skip = 1;       % put a number of blocs which you want to skip from processing
N_blocks_to_process = 65;    % how many data-blocs to process

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
% plot(B)
% xlabel('Fast time, msec')
% title('ADC1, Rx')


%% split into fast-time and slow-time
starting_point = 200962; 
slowtime_length = 399994;                       % TODO:: in real data it is around 399994.3
slowtime = floor(N_blocks_to_process/2);
end_point = starting_point + slowtime_length * slowtime - 1;
received = B(starting_point:end_point);
transmitted = A(starting_point:end_point);
% clear B;
data = reshape(received, [slowtime_length, slowtime]);
tdata = reshape(transmitted, [slowtime_length, slowtime]);

figure()
imagesc(tdata)
colorbar()
colormap(jet)

% effective_length
effective_length = 400000;

% save('data.mat', 'data', 'tdata')

%% lowpass filter
% spectra
f = (F_sampling/MHz)*linspace(0,1,N_sample);
W = ones(N_sample, 1);
center_frequency = 125e6;
dems = exp(-2j*pi*center_frequency*[0:(length(A)-1)]'/F_sampling * 1);

y1=fft(W.*A.*dems, max(size(A)));
z1 = (fftshift(y1));

y2=fft(W.*B.*dems,max(size(B)));
z2 = (fftshift(y2));

f = (F_sampling/MHz)*linspace(-0.5,0.5,N_sample);
frequency=f((1:floor(N_sample)));
figure;
subplot(2,1,1)
plot(frequency,db(z1));
xlabel('Frequency')
title('Spectrum ADC0, Tx')

subplot(2,1,2)
plot(frequency,db(z2));
xlabel('Frequency')
title('Spectrum ADC1, Rx')

% lowpass filter
freq_lengtg = length(z1);
zero_length = floor(freq_lengtg/4);

Z1 = z1;
Z1(1:zero_length) = 0;
Z1(3*zero_length:end)=0;
figure()
plot(db(Z1))
a = ifft(ifftshift(Z1));

figure()
plot(real(a(200962: (200962+196608))))

Z2 = z2;
Z2(1:zero_length) = 0;
Z2(3*zero_length:end)=0;
figure()
plot(db(Z2))
b = ifft(ifftshift(Z2));
figure()
plot(real(b(200962: (200962+196608))))

%% Load codes

T = readtable('pmcw_waveform.txt');
wave = T{:, 1};
wave = wave(1:3:end);
wave_length = size(wave);

wavefft = fftshift(fft(wave.* exp(-2j*pi*center_frequency*[0:(length(wave)-1)]'/F_sampling * 1)));
W = wavefft;
W(1:100000) = 0;
W(3*100000:end)=0;
w = ifft(ifftshift(W));
figure()
plot(db(W))
    
    
%% Circular Correlation
starting_point = 200962; %  600956 (second starting point)
end_point = 200962+262139;

wave_ = w(262139:-1:1);
wave__ = w(1:262139);
CORRB = cconv((b(starting_point:end_point)), (wave_), 262139);

CORRB_length = length(CORRB);

%% Figure
dr = c/2/F_sampling;
x = 0:dr:dr*262138;

figure
plot(x(1:12000), db(CORRB(1:12000)))
xlabel('Distance (m)')
ylabel('dB')
grid on





































    
    
    