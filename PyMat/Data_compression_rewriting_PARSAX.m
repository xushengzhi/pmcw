clc;
clear all;
image_on = 0;

DataDir='D:\2014-11-28 A13 Nikita\';
% Noise file
%filenaam = [DataDir, 'N_VV_20141128131313.bin'];   Tr = 1e-3;   % 1 kHz 
% Data files
filenaam = [DataDir, 'VV_20141128140348.bin'];   Tr = 1e-3;   % 1 kHz
%filenaam = [DataDir, 'VV_20141128140657.bin'];   Tr = 0.5e-3;; % 2 kHz

M = 128; 
m = (0:M-1).';
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

% t_mat = repmat(t, M, 1);
% m_mat = repmat(m, 1, Nt);
% t_full = t_mat + m_mat*Tr_z;
clear m_mat;

%% LRR
LRR_range_cells = 32;
LRR_cen_range = 4000;

%% Coherent integartion
Ambig = 3;
vaxe_amb = -Va*Ambig/2 : Va/Nv : Va*Ambig/2-Va/Nv; 
Nv_ambig = length(vaxe_amb);

% Read file
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

N_samples_PRI = Tr_z * Fs;

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
    A=[A(:); B(:)]; % 
end;
clear B;
fclose(fid);

Time_read = size(A, 1) / 2 / Fs;

% At this point all avalable data are in variable A, B
N_sample=max(size(A));

% reshape data in two channeles 
A=reshape(A, 2, N_sample/2);
s_tx_IF = A(1,:);       
s_rx_IF = A(2,:);
clear A;

%% Find the start to read
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
N_sample_proc = pos(M + 1) - pos(1); 

t = (0: N_sample_proc-1) /Fs;
S_IF = exp(-1j*2*pi*F_IF*t);
clear t;

s_tx_IF_M = zeros(M, N_samples_PRI);
s_rx_IF_M = zeros(M, N_samples_PRI);
s_IF_M = zeros(M, N_samples_PRI);

for pulse = 1:M 
    s_tx_IF_M(pulse, 1:(pos(pulse+1)-pos(pulse))) = s_tx_IF(pos(pulse) : pos(pulse+1)-1); %  for TX read one pulse anyway!!
    s_rx_IF_M(pulse, 1:(pos(pulse+1) -pos(pulse))) = s_rx_IF(pos(pulse) : pos(pulse+1)-1);
    s_IF_M(pulse, 1:(pos(pulse+1)-pos(pulse))) = S_IF((pos(pulse) : pos(pulse+1)-1) - pos(1) + 1);
end;

clear S_IF
clear s_tx_IF
clear s_rx_IF
clear S_IF_M

%% signal

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

clear s_tx_I
clear s_tx_IF
clear s_tx_Q

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

clear s_rx_I
clear s_rx_IF
clear s_rx_Q


%% Deramping
%{
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
%}
% LRR
%{
LRR_beg_meters = LRR_cen_range - LRR_range_cells/2 * dR_deramp;
LRR_beg_meters = floor(LRR_beg_meters / dR_deramp) * dR_deramp;
LRR_beg = floor(LRR_beg_meters / dR_deramp);
LRR_cells_deram = LRR_beg : (LRR_beg+LRR_range_cells)-1;

raxe_LRR_deram = raxe_deram(LRR_cells_deram);
r_st = r(:, LRR_cells_deram).';
clear r;
clear raxe_deram;

Nf = length(LRR_cells_deram);
ff_st = fft(r_st, Nf, 1);

% Coherent Integration

mu_1 = BW_deramp / (Fc-BW/2) / Nf;
Y_RV_WB = Fast_CI(ff_st, Ambig, Va, dv, vel_upsampl, range_upsampl, mu_1, vaxe_amb, 1);

figure(11)
colormap(flipud(hot))
subplot(2,1,1);
imagesc(vaxe_amb, raxe_LRR_deram, 20*log10(abs(Y_RV_WB)))
max_I = max(max(20*log10(abs(Y_RV_WB))));
caxis([max_I - DB_Dif, max_I])
ylabel('Range')
xlabel('Velocity, m/s')
title('Deramping')
colorbar;
%}

%% Mathced filter
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

clear s_rx_fft
clear s_tx_fft
clear s_r_fft

% Normalisation on the noise floor
load var_MF
r_mf = r_mf / (var_MF)^0.5;
%% Wtite to file
%
filename = strcat('Compressed data at_', num2str(1/(Tr*1000),'%10.2f'), ' kHz' , '_mf_', ...
        num2str(M),'_pulses_hamming_tapering_', num2str(N_r_MF), 'range_cells_new');
Data_FullName = strcat('D:\PARSAX_RAW_DATA_PROCESSING\Range_compression\', filename, '.bin');
Data_fid = fopen(Data_FullName, 'w');
if (Data_fid == -1)
    error('Cannot open file');
end
size_r = [size(r_mf, 1) , size(r_mf, 2)];
fwrite(Data_fid, size_r , 'int16');
r_compl = [real(r_mf(:)), imag(r_mf(:))];
count = fwrite(Data_fid, r_compl , 'double')
fclose(Data_fid);

clear r_mf;
clear r_compl;
%}

% LRR
%{
LRR_beg_meters = LRR_cen_range - LRR_range_cells/2 * dR;
LRR_beg_meters = floor(LRR_beg_meters / dR) * dR;
LRR_beg = floor(LRR_beg_meters / dR);
LRR_cells = LRR_beg : (LRR_beg+LRR_range_cells)-1;

r_st_2 = r_mf(:, LRR_cells).';

Nf = length(LRR_cells);
raxe_LRR = raxe_MF(LRR_cells);
clear raxe

ff_st_2 = fft(r_st_2, Nf, 1);
ff_st_2 = fftshift(ff_st_2 , 1);

mu_1 = BW / (Fc-BW/2) / Nf;
Y_RV_WB_2 = Fast_CI(ff_st_2, Ambig, Va, dv, vel_upsampl, range_upsampl, mu_1, vaxe_amb, 1);

figure(11)
subplot(2,1,2);
imagesc(vaxe_amb, raxe_LRR, 20*log10(abs(Y_RV_WB_2)))
max_I = max(max(20*log10(abs(Y_RV_WB_2))));
caxis([max_I - DB_Dif, max_I])
ylabel('Range')
xlabel('Velocity, m/s')
title('Matched filter')
colorbar;
%}

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
