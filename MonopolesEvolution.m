load data/GS_phi
%% Paramemters in Simulation
% DESCRIPTIVE TEXT
hbar = 1.05e-34; mu_B=9.27e-24;

N = 1.82e5; m = 1.443e-25; a0 = 5.387e-9; a2 = 5.313e-9; 
omega_r = 2*pi*160; omega_z = 2*pi*220;
bq = 3.7e-2;g_F=-1/2;

a = sqrt(hbar/(m*omega_r)); 
beta0 = 4*pi*N*(a0+2*a2)/3/a;  beta2 = 4*pi*N*(a2-a0)/3/a;
gamma_x = 1; gamma_y = 1; gamma_z = omega_z/omega_r;
q1 = mu_B*g_F/(hbar*omega_r); q2 = 2*pi*hbar*70e-4/(hbar*omega_r);
% beta0 = 1000; beta2 = -26; 
b = bq*a;
%% Setup grid and initial state
% gird in real space
n = 128;
h = 0.24;
x=(-n/2:n/2-1)*h;
[x,y,z]=meshgrid(x);

% grid in momentum space
hw=2*pi/(n*h);
wx=fftshift(-n/2:n/2-1)*hw;
[wx,wy,wz]=meshgrid(wx);

% % intial wavefunction
% phi1=sqrt(gamma_z)/pi^(3/4)*exp(-(x.^2+y.^2+(gamma_z*z).^2)/2)/sqrt(3);
% phi2=sqrt(gamma_z)/pi^(3/4)*exp(-(x.^2+y.^2+(gamma_z*z).^2)/2)/sqrt(3);
% phi3=sqrt(gamma_z)/pi^(3/4)*exp(-(x.^2+y.^2+(gamma_z*z).^2)/2)/sqrt(3);

% normalizing
P = h^3*sum(sum(sum(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)));

% setup magnetic field B=(0,0,1) 
Bz0 = 10e-7; %10mG
Bx=b*x;
By=b*y;
% Bz=-2*b*z+Bz0;

%% main loops
maxt = 50;  % 演化50ms
ht=0.005;
maxstep=maxt/ht;    


V=1/2*(x.^2+y.^2+(gamma_z*z).^2);                         % potential in real space
D=1/2*(wx.^2+wy.^2+wz.^2);                      % potential in momentum space

V=gpuArray(single(V));
D=gpuArray(single(D));
phi1=gpuArray(single(phi1));
phi2=gpuArray(single(phi2));
phi3=gpuArray(single(phi3));
Bx=gpuArray(single(Bx));
By=gpuArray(single(By));
z=gpuArray(single(z));


    
%% use Runge-Kutta method of four order
nSx =@(phi1,phi2,phi3) real(conj(phi1).*phi2+conj(phi2).*(phi1+phi3)+conj(phi3).*phi2)/sqrt(2);
nSy =@(phi1,phi2,phi3) imag(-conj(phi1).*phi2+conj(phi2).*(phi1-phi3)+conj(phi3).*phi2)/sqrt(2);
nSz =@(phi1,phi2,phi3) real(conj(phi1).*phi1-conj(phi3).*phi3);
Bz = @(t) -2*b*z+Bz0-0.25e-4*t/omega_r; % 引入含时Bz


f1=@(phi1,phi2,phi3,t) -1i*(V+beta0*(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)).*phi1+...
    -1i*(q1*Bz(t)+beta2*nSz(phi1,phi2,phi3)+q2/2*(2*Bz(t).^2+Bx.^2+By.^2)).*phi1+...
    -1i*(q1/sqrt(2)*(Bx-1i*By)+beta2/sqrt(2)*(nSx(phi1,phi2,phi3)-1i*nSy(phi1,phi2,phi3))+q2/2*sqrt(2)*Bz(t).*(Bx-1i*By)).*phi2+...
    -1i*(0+q2/2*(Bx-1i*By).^2).*phi3;


f2=@(phi1,phi2,phi3,t) -1i*(V+beta0*(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)).*phi2+...
    -1i*(q1/sqrt(2)*(Bx+1i*By)+beta2/sqrt(2)*(nSx(phi1,phi2,phi3)+1i*nSy(phi1,phi2,phi3))+q2/2*sqrt(2)*Bz(t).*(Bx+1i*By)).*phi1+...
    -1i*(0+q2/2*sqrt(2)*(Bx.^2+By.^2)).*phi2+...
    -1i*(q1/sqrt(2)*(Bx-1i*By)+beta2/sqrt(2)*(nSx(phi1,phi2,phi3)-1i*nSy(phi1,phi2,phi3))-q2/2*sqrt(2)*Bz(t).*(Bx-1i*By)).*phi3;


f3=@(phi1,phi2,phi3,t) -1i*(V+beta0*(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)).*phi3+...
    -1i*(0+q2/2*(Bx+1i*By).^2).*phi1+...
    -1i*(q1/sqrt(2)*(Bx+1i*By)+beta2/sqrt(2)*(nSx(phi1,phi2,phi3)+1i*nSy(phi1,phi2,phi3))-q2/2*sqrt(2)*Bz(t).*(Bx+1i*By)).*phi2+...
    -1i*(-q1*Bz(t)-beta2*nSz(phi1,phi2,phi3)+q2/2*(2*Bz(t).^2+Bx.^2+By.^2)).*phi3;

tic
for nstep=0:maxstep
    t=nstep*ht;
    
    per = nstep/maxstep*100;
    if mod(per,0.5)==0
        disp(['已经完成',num2str(per),'%']);
    end
%     nstep 
    phi1temp=phi1;
    phi2temp=phi2;
    phi3temp=phi3;
    
    % solve the part of the equation in momentum space (use FFT)
    phi1 = ifftn(exp(-1i*D*ht/2).*fftn(phi1));
    phi2 = ifftn(exp(-1i*D*ht/2).*fftn(phi2));
    phi3 = ifftn(exp(-1i*D*ht/2).*fftn(phi3));
    
    % use Runge-Kutta method of four order
    K11 = f1(phi1,phi2,phi3,t);
    K12 = f2(phi1,phi2,phi3,t);
    K13 = f3(phi1,phi2,phi3,t);
    
    K21 = f1(phi1+ht*K11/2, phi2+ht*K12/2, phi3+ht*K13/2,t+ht/2);
    K22 = f2(phi1+ht*K11/2, phi2+ht*K12/2, phi3+ht*K13/2,t+ht/2);
    K23 = f3(phi1+ht*K11/2, phi2+ht*K12/2, phi3+ht*K13/2,t+ht/2);
    
    K31 = f1(phi1+ht*K21/2, phi2+ht*K22/2, phi3+ht*K23/2,t+ht/2);
    K32 = f2(phi1+ht*K21/2, phi2+ht*K22/2, phi3+ht*K23/2,t+ht/2);
    K33 = f3(phi1+ht*K21/2, phi2+ht*K22/2, phi3+ht*K23/2,t+ht/2);
    
    K41 = f1(phi1+ht*K31, phi2+ht*K32, phi3+ht*K33,t+ht);
    K42 = f2(phi1+ht*K31, phi2+ht*K32, phi3+ht*K33,t+ht);
    K43 = f3(phi1+ht*K31, phi2+ht*K32, phi3+ht*K33,t+ht);   
    
    phi1 = phi1+ht*(K11+2*K21+2*K31+K41)/6;
    phi2 = phi2+ht*(K12+2*K22+2*K32+K42)/6;
    phi3 = phi3+ht*(K13+2*K23+2*K33+K43)/6;
    
    % again FFT
    phi1 = ifftn(exp(-1i*D*ht/2).*fftn(phi1));
    phi2 = ifftn(exp(-1i*D*ht/2).*fftn(phi2));
    phi3 = ifftn(exp(-1i*D*ht/2).*fftn(phi3));    
    
% 在30-50ms区间，每1ms输出数据
if t>=30 && mod(t,1)==0
    phi1_tmp=gather(phi1);
    phi2_tmp=gather(phi2);
    phi3_tmp=gather(phi3);
    filename = ['data/phis_',num2str(t),'ms'];
    save(filename,'phi1_tmp','phi2_tmp','phi3_tmp')
end
    
end
toc


