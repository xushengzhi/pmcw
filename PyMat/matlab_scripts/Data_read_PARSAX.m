clc;
clear all;
image_on = 0;

DataDir='L:\me\MS3-PARSAX\2014\2014-11-28 A13 Nikita\';
% Noise file
%filenaam = [DataDir, 'N_VV_20141128131313.bin'];   Tr = 1e-3;   % 1 kHz 
% Data files
filenaam = [DataDir, 'VV_20141128140348.bin'];   Tr = 1e-3;   % 1 kHz
%filenaam = [DataDir, 'VV_20141128140657.bin'];   Tr = 0.5e-3;; % 2 kHz

M = 8; 
m = (0:M-1).';
% Tr = 1e-3;
BW = 100e6;
F_IF = 125e6;
beta = BW / Tr;
c = 3e8;
Fc = 3.315e9;
dR = c / (2*BW);

N_r_MF = 4* 1024; % 1024; %8192;   % Number of range cells in the area of interest
R_max_deram = N_r_MF * dR;
f_b_max = N_r_MF / Tr;

Fs = 400e6;
dt = 1/ Fs;
% t = 0: dt : Tr - dt;
% Nt = length(t);
range_upsampl = 2; %[1, 2, 4]
DB_Dif = 40;

zero_time = 20e-6;
Tr_z = Tr + zero_time;
t = 0: dt : Tr_z - dt;
Nt = length(t);

L0 = c / (Fc -BW/2);
Va = L0 / (2*Tr_z);
migr = Va * M * Tr_z / dR
vel_upsampl = 4;
Nv = vel_upsampl*M;
dv = Va/Nv;
vaxe = -Va/2 : dv : Va/2-dv; 

t_mat = repmat(t, M, 1);
m_mat = repmat(m, 1, Nt);
t_full = t_mat + m_mat*Tr_z;
clear m_mat;





%%
%% =======================================
% Read file
%% ================================== 
fid=fopen(filenaam)
info=fread(fid, 2,'int32'); % Header (2*int32) contain the size of data block in int32 with header  
fseek(fid, 0, 'bof');
samples_per_block_int32 = (info(1)-2); % in int32 without header
int_32_size = 4; %byte
int_16_size = 2; %byte
channels = 2; % tx and rx
samples_per_block = samples_per_block_int32 * int_32_size/ int_16_size / 2; % Sampled in int16 in each channel

Time_to_read = (M + 3) * Tr; % +2 beacause of the cutting the first and the last

N_blocks_to_skip = 2; % put a number of blocs which you want to skip from processing
N_blocks_to_process = ceil(Time_to_read / (samples_per_block / Fs));  % how many data-blocs to process  

N_samples_PRI = round(Tr_z * Fs);

if (N_blocks_to_skip > 0)       % Skip blocks
    for i=1:N_blocks_to_skip
        A = fread(fid, info(1),'int32');
    end;
end;

A= []; B = [];
inf = zeros(N_blocks_to_process, 2);
for i=1:N_blocks_to_process
    inf(i, :) = fread(fid, 2,'int32');
    B = fread(fid, inf(i)*2 - 4,'int16');
    %B(1:4)=[];  % remove header
    A=[A(:); B(:)]; % 
end;
clear B;
fclose(fid);

Time_read = size(A, 1) / 2 / Fs;

%==================================
% At this point all avalable data are in variable A, B
%==================================
N_sample=max(size(A));

% reshape data in two channeles 
A=reshape(A, 2, N_sample/2);
s_tx_IF = A(1,:);       
s_rx_IF = A(2,:);

clear A;


%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find the start to read
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Thresh = 1000; % assume that values lower is noise only - to detect the begining of the pulse in tx
leng = 500; % min number of samples less then threshold to find
pos = zeros(M + 1, 1);
prev_pulse_start = 0;

for pulse = 1: M + 1  % +1 to read the begining of the first pulse out of interest
    a = abs(s_tx_IF(prev_pulse_start+1 : prev_pulse_start + N_samples_PRI*1.5)) > Thresh; %  0 corresp to pause
    positions = findstr(a,zeros(1,leng)); % FIND THE FIRST ZEROS
    pos_in_piece = max(positions) + leng - 1;
    pos(pulse) = prev_pulse_start + pos_in_piece ; % sample of each oulse starts
    prev_pulse_start = pos(pulse);
end;
clear a;
clear positions;


%% Positioning pulses equspasely
N_sample_proc = pos(M + 1) - pos(1); %  +2 - till the ent of the next pulse
% 
% pos = pos(1) + floor(N_sample_proc / (M ) * (0:M+1));
% leng_2(1: M ) = pos(2: M+1 ) - pos(1: M);

t = (0: N_sample_proc-1) /Fs;
S_IF = exp(-1j*2*pi*F_IF*t);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Add sinthetic target
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
v_0 = 50; %m/s;
R_0 = 4000; %m
a_0 = 10; %0.01;
tau_0 = 2*R_0 / c_speed;

dt = 1 /F_sampling;
tf = (0: (fast_time_leng -1))* dt;
Nt = length(tf);
t_1 = [tf, tf+Nt*dt];
m = (0:Pulses - 1).';
t_mat_1 = repmat(t_1, Pulses, 1);
Nt_non_zeros = PRI_non_zeros * F_sampling;
m_mat = repmat(m, 1, Nt_non_zeros);
m_mat_1 = repmat(m, 1, 2*Nt);

rect_pulse_0 = (t_mat_1 >= tau_0) & (t_mat_1 < PRI_non_zeros+tau_0); %(t_mat_1 < Nt*dt+tau_0);
ind_beg_pulse = find(t_1 >= tau_0, 1);
ind_end_pulse = find(t_1 >= PRI_non_zeros+tau_0, 1)-1;
t_mat_2 = t_mat_1(:, ind_beg_pulse :ind_end_pulse );
t_full_2 = t_mat_2 + m_mat * Nt*dt;

s_rx_IF_ss = zeros(Pulses, ind_end_pulse - ind_beg_pulse + 1);

s_rx_IF_ss(:, :) = a_0 * exp(2*pi*1j* beta/2 *(t_mat_2 - tau_0 + 2*v_0/c_speed * t_full_2 ).^2) ...
    .* exp(2*pi*1j * (F_IF -BW/2) * t_full_2)... 
    .* exp(-2*pi*1j * (fc - BW/2) * (2*v_0/c_speed * t_full_2 - tau_0));

s_rx_IF_s = zeros(1, (Pulses+1)*Nt);
for pulse = 1:Pulses
    s_rx_IF_s((pulse-1)*Nt +ind_beg_pulse  : (pulse-1)*Nt + ind_end_pulse) =  s_rx_IF_ss(pulse, :) + s_rx_IF_s((pulse-1)*Nt +ind_beg_pulse  : (pulse-1)*Nt + ind_end_pulse) ;
end;
s_rx_IF_s = real(s_rx_IF_s) ;

s_rx_IF(pos(1):pos(Pulses+2)-1) = s_rx_IF_s + s_rx_IF(pos(1):pos(Pulses+2)-1) ;
%}
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s_tx_IF_M = zeros(M, N_samples_PRI);
s_rx_IF_M = zeros(M, N_samples_PRI);
s_IF_M = zeros(M, N_samples_PRI);

for pulse = 1:M 
    s_tx_IF_M(pulse, 1:(pos(pulse+1)-pos(pulse))) = s_tx_IF(pos(pulse) : pos(pulse+1)-1); %  for TX read one pulse anyway!!
    s_rx_IF_M(pulse, 1:(pos(pulse+1) -pos(pulse))) = s_rx_IF(pos(pulse) : pos(pulse+1)-1);
    s_IF_M(pulse, 1:(pos(pulse+1)-pos(pulse))) = S_IF((pos(pulse) : pos(pulse+1)-1) - pos(1) + 1);
end;







S_IF_I = imag(s_IF_M);   %  -sin 
S_IF_Q = real(s_IF_M);  % cos
clear s_IF_M;

s_tx_I = s_tx_IF_M .* S_IF_I; %cos(2*pi*F_IF*t_mat);
s_tx_Q = s_tx_IF_M .* S_IF_Q; %(-sin(2*pi*F_IF*t_mat));
clear s_tx_IF_M;

% Filter design
Fcut = 75e6; % cutoff frequency
N  = 100;  % FIR filter order
LP = fdesign.lowpass('N,Fc',N,Fcut,Fs); % Fs is always trailing argument
FIR_BB = design(LP,'window','window',@hamming);

s_tx_I = filter(FIR_BB, s_tx_I, 2);
s_tx_Q = filter(FIR_BB, s_tx_Q, 2);

s_tx_I = downsample(s_tx_I.',2).';
s_tx_Q = downsample(s_tx_Q.',2).';
Fs = Fs/2;
Nt = Nt/2;

s_tx_BB = s_tx_I + 1i*s_tx_Q;

if image_on
    subplot(3,1,3);
    plot(-Fs/2: Fs/Nt: Fs/2-Fs/Nt, 20*log10(abs( fftshift(fft(s_tx_BB, [], 2)))));
    title('Complex BB')
end;



%% Periodogramm
%
if  image_on
    s_t = s_tx_BB.';
    s_t = s_t(:);
    % setting parameters for STFFT
    fft_length = 512;
    window = ones(1, fft_length); % no tapering
    nfft = 4*length(window);
    noverlap = 3/4 * fft_length;

    [S,F,T,P] = spectrogram(s_t , window, noverlap, nfft, Fs/2);
    figure(7); 
    surf(T*1e3, F*1e-6,10*log10(P),'edgecolor','none'); axis tight;
    %surf(10*log10(P),'edgecolor','none'); axis tight; 
    view(0,90);
    xlabel('Time [ms]'); ylabel('MHz');
    clear s_t
end;
%}
if  image_on
    figure(2);
    subplot(3,1,1);
    plot(t, real(s_rx_IF.'));
    title('Recieved signal IF')

    subplot(3,1,2);
    plot(-Fs/2: Fs/Nt: Fs/2-Fs/Nt, fftshift(20*log10(abs(fft(s_rx_IF, [], 2)))) );
    title('Recieved  spectrum IF')
end;


clear S_IF
clear s_tx_IF
clear s_rx_IF
clear S_IF_M

clear s_tx_I
clear s_tx_IF
clear s_tx_Q


clear t;


s_rx_I = s_rx_IF_M .* S_IF_I; %cos(2*pi*F_IF*t_mat);
s_rx_Q = s_rx_IF_M .* S_IF_Q; %(-sin(2*pi*F_IF*t_mat));
clear S_IF_I;
clear S_IF_Q;
clear s_rx_IF_M

s_rx_I = filter(FIR_BB, s_rx_I, 2);
s_rx_Q = filter(FIR_BB, s_rx_Q, 2);

s_rx_I = downsample(s_rx_I.',2).';
s_rx_Q = downsample(s_rx_Q.',2).';


s_rx_BB = s_rx_I + 1i*s_rx_Q;

if image_on
    figure(2);
    subplot(3,1,3);
    plot(-Fs/2: Fs/Nt: Fs/2-Fs/Nt, 20*log10(abs( fftshift(fft(s_rx_BB, [], 2)))));
    title('Complex BB')
end;

clear s_rx_I
clear s_rx_IF
clear s_rx_Q


%% Periodogramm
%
if image_on

    s_r = s_rx_BB.';
    s_r = s_r(:);
    % setting parameters for STFFT
    fft_length = 512;
    window = ones(1, fft_length); % no tapering
    nfft = 4*length(window);
    noverlap = 3/4 * fft_length;

    [S,F,T,P] = spectrogram(s_r , window, noverlap, nfft, Fs);
    figure(8); 
    surf(T*1e3, F*1e-6,10*log10(P),'edgecolor','none'); axis tight;
    %surf(10*log10(P),'edgecolor','none'); axis tight; 
    view(0,90);
    xlabel('Time [ms]'); ylabel('MHz');
    clear s_r
end;
%}

%
%% Deramping
%
s_mix = s_rx_BB(:, 1:Nt) .* conj(s_tx_BB(:, 1:Nt)) ;

tau_max = 2 * R_max_deram /c;
BW_deramp = BW * (1 - tau_max/Tr);
dR_deramp = c / (2*BW_deramp);
Nr_deramp = floor(R_max_deram / dR_deramp);
raxe_deram = (0 : Nr_deramp - 1) * dR_deramp;

% remove the piece then tau < tau_0    
tau_points_beg = ceil(tau_max * Fs);   
tau_points_end = floor(Tr * Fs); % remove zeros at the end
s_mix = s_mix(:, tau_points_beg : tau_points_end); 
    
% Filter design
Fcut = f_b_max; % cutoff frequency

N  = 500;  % FIR filter order
LP = fdesign.lowpass('N,Fc',N,Fcut,Fs); % Fs is always trailing argument
FIR_beat = design(LP,'window','window',@hamming);

s_mix = filter(FIR_beat, s_mix, 2);
downsampling_deramping = floor(Fs / f_b_max  );
s_mix = downsample(s_mix.', downsampling_deramping ).' * downsampling_deramping;

% Tapering
Window_dr = hamming(size(s_mix, 2));
s_mix = s_mix .* repmat(Window_dr.', M, 1) /  (sum(Window_dr) / size(s_mix, 2));


r = fft(s_mix, size(s_mix, 2), 2);
clear s_mix;
% r = r(:, [1, end:-1:2]);



if  image_on
    figure(3);
    plot(raxe_deram, 20*log10(abs(r.')));
    title('Range profile - Deramping')
end;


% Doppler processing deramping
rv = fftshift(fft(r, Nv, 1), 1);
figure(4);
imagesc(vaxe, raxe_deram, 20*log10(abs(rv.')))
colormap(flipud(hot))
 ylabel('Range, m')
 xlabel('Velocity, m/s')
 title('Deramping')
 colorbar
 axis xy;






%% Mathced filter
%
raxe_MF = (0:N_r_MF - 1) * dR;

% Tapering
Window_mf = hamming(Nt);
s_rx_BB = s_rx_BB .* repmat(Window_mf.', M, 1) / (sum(Window_mf) / Nt);
clear Window_mf;

s_rx_fft = fft([s_rx_BB, zeros(size(s_rx_BB))] , [], 2);
s_tx_fft = fft([s_tx_BB, zeros(size(s_rx_BB))], [], 2);

s_r_fft = s_rx_fft .* conj(s_tx_fft);


r_mf = ifft(s_r_fft, [], 2);

r_mf = downsample(r_mf.', Fs/BW ).';

r_mf = r_mf(:, 1:N_r_MF);

if  image_on
    figure(55);
    plot(20*log10(abs(r_mf.')));
    title('Range profile - Matched filter')
end;


% Doppler processing matched filter
rv_mf = fftshift(fft(r_mf, Nv, 1), 1);
figure(5)
colormap(flipud(hot))
imagesc(vaxe, raxe_MF, 20*log10(abs(rv_mf.')))
ylabel('Range, m')
xlabel('Velocity, m/s')
title('Matched filter')
colorbar
clear rv_mf

    
clear s_rx_fft
clear s_tx_fft
clear s_r_fft



%% Clear variavles
clear s_rx_BB
clear s_tx_BB
clear t
clear t_full
clear t_mat
clear m_mat

clear r_down;
clear r_mf

script_finished
