% =========================================================================
% Script for the analysis of the X5-400M direct-streamed binary-files 
% (via original SNAP program)
% =========================================================================
% created 18.06.2009
% updated 07.10.2014
% updated 16.09.2019
% =========================================================================
% Important notes:
% - In this streaming mode there is no any added feature into the signal
% for timing/structure specification - streaming started by pressing buttom
% and stoped as soon as selectet in SNAP amoiut of data collected. No any
% internal triggering related to signal structure.
% - it means that it is responcibility of the signal's author to find
% predefined features in signals or defime signal's periodicity with any
% other machnisms (e.g. to  count samples till repetitive signals structures)
% - block size in this mode relates only to the operation of data transfer
% from the board to the host computer memory
% - it is impossible in this mode to guaranty the coherency/synchronization 
% of the acquired data between different boards.
% - after 300 MB of collected data may take place uncontrollable and 
% untraceble losses of data (waveforms becomes shorter)
% - any other questions and problems - come and ask...
% =========================================================================
% o.a.krasnov@tudelft.nl, IRCTR/MTSR/MS3, TU Delft
% =========================================================================
close all
clear all
clc

% put your data dir here
% DataDir='K:\me\MS3\Studenten\Shengzhi Xu\HH';
% cd(DataDir);

% dialog to select data-file
% file = 'HH_20190916141206.bin';
[filenaam,DataDir]=uigetfile('*.bin');
filenaam=[DataDir,filenaam];

% =========================================================================
% constants
pi2=2*pi;
MHz=1000000;   % MHz to Hz conversion
msec=0.001;    % msec to sec conversion
usec=0.000001; % usec to sec conversion

% =========================================================================
F_sampling = 400*MHz;   % Sampling frequency
N_sample=17000000;      % number of samples
two_channel=1;          % 1 if two channels presented in data file

% =========================================================================
% read file
% =========================================================================
% reading the header of the first block (and assume that all the following
% blocks are the same. This is usually correct)

fid=fopen(filenaam)
info=fread(fid, 2,'int32')
fseek(fid, 0, 'bof');

%=======================================
%Datafile organization:
% X5-400M data-blocks (~2MB per 2 channels) =
% ~ 512 kSamples per block per channel
%=======================================
N_blocks_to_skip = 1; % put a number of blocs which you want to skip from processing
N_blocks_to_process = 5; % how many data-blocs to process 

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

%==========================================================================
% At this point all avalable data are in variable A
%==========================================================================
N_sample=max(size(A))

if (two_channel==1)
    % reshape data in two channeles 
    B=reshape(A,2,N_sample/2);
    A=B(1,:);
    B=B(2,:);
    A=A(:);
    B=B(:);
    N_sample=max(size(A))
    
%     N_sample=200000;
    A=A(1:N_sample);
    B=B(1:N_sample);
   

    %time array for plotting
    t=(1/400/1000).*linspace(0,N_sample,N_sample+1); % msec
    t(end)=[];

    % =====================================================================
    % ploting of time seria data from two channels
    % =====================================================================
    figure;
    subplot(2,1,1)
%     plot(t(1:50000),A(1:50000));
    plot(A)
    xlabel('Fast time, msec')
    title('ADC0, Tx')

    subplot(2,1,2)
%     plot(t(1:50000),B(1:50000));
    plot(B)
    xlabel('Fast time, msec')
    title('ADC1, Rx')

%     % =====================================================================
%     % ploting of spectral data from two channels
%     % =====================================================================
    %frequency grid for further visualization
    f = (F_sampling/MHz)*linspace(0,1,N_sample);
    frequency=f((1:floor(N_sample/2)));
    % windowing function for fft
%     W=hamming(N_sample);
    W = ones(N_sample, 1);
%%     
    center_frequency = 125e6;
    dems = exp(-2j*pi*center_frequency*[0:(length(A)-1)]'/F_sampling * 1);
    
    % fft itself
     y1=fft(W.*A.*dems, max(size(A)));
     z1 = (fftshift(y1));
%      z1=db(y1(1:floor(N_sample/2)));
     y2=fft(W.*B.*dems,max(size(B)));
     z2 = (fftshift(y2));
%      z2=db(y2(1:floor(N_sample/2)));
% Spectra
% 
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
 %%     
    figure
    plot(db(z2.*conj(z1)))
end

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
%%

T = readtable('pmcw_waveform.txt');
wave = T{:, 1};
wave = wave(2:3:end);
wave_length = size(wave);
% COR = zeros(N_sample - wave_length(1), 1);

wavefft = fftshift(fft(wave.* exp(-2j*pi*center_frequency*[0:(length(wave)-1)]'/F_sampling * 1)));
% figure()
% plot(db(wavefft))
% W = zeros(length(wavefft), 1);
W = wavefft;
W(1:100000) = 0;
W(3*100000:end)=0;
w = ifft(ifftshift(W));
figure()
plot(real(w))

%% using complex number
starting_point = 200962; %  600956 (second starting point)
end_point = 200962+262139;

wave_ = w(262139:-1:1);
wave__ = w(1:262139);
% tic
% CORRA = conv(a(starting_point:end_point), wave_, 'same');
% toc
CORRB = cconv((b(starting_point:end_point)), (wave_), 262139);
% CORRB = cconv(B(starting_point:end_point), wave(262139:-1:1), 262139);
CORRB_length = length(CORRB);
c = 3e8;
dr = c/2/F_sampling;
x = 0:dr:dr*262138;

%% 
figure
plot(x, db(CORRB))
xlabel('Distance (m)')
ylabel('dB')
grid on
% hold on
% plot(x, db(CORRA))
% legend('R', 'T')

















