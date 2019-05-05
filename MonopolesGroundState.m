%% Paramemters in Simulation
% DESCRIPTIVE TEXT
hbar = 1.05e-34; mu_B=9.27e-24;

N = 1.82e5; m = 1.443e-25; a0 = 5.387e-9; a2 = 5.313e-9; 
omega_r = 2*pi*160; omega_z = 2*pi*220;
bq = 3.7e-2;g_F=-1/2;


a = sqrt(hbar/(m*omega_r)); 
beta0 = 4*pi*N*(a0+2*a2)/3/a;  beta2 = 4*pi*N*(a2-a0)/3/a;
gamma_x = 1; gamma_y = 1; gamma_z = omega_z/omega_r;
q1 = mu_B*g_F/(hbar*omega_r);
q2 = 2*pi*hbar*70e8/(hbar*omega_r);
% beta0 = 1000; beta2 = -26; 
b = bq*a;

%% Setup grid and initial state
% gird in real space
n=128;
h = 0.24;
x=(-n/2:n/2-1)*h;
[x,y,z]=meshgrid(x);

% grid in momentum space
hw=2*pi/(n*h);
wx=fftshift(-n/2:n/2-1)*hw;
[wx,wy,wz]=meshgrid(wx);

% intial wavefunction
phi1=sqrt(gamma_z)/pi^(3/4)*exp(-(x.^2+y.^2+(gamma_z*z).^2)/2)/sqrt(3);
phi2=sqrt(gamma_z)/pi^(3/4)*exp(-(x.^2+y.^2+(gamma_z*z).^2)/2)/sqrt(3);
phi3=sqrt(gamma_z)/pi^(3/4)*exp(-(x.^2+y.^2+(gamma_z*z).^2)/2)/sqrt(3);

% normalizing
P = h^3*sum(sum(sum(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)));

% setup magnetic field B=(0,0,1) 
Bz0 = 10e-7;  %10mG
Bx = b*x;
By = b*y;
% Bz = -2*b*z+Bz0;
Bz = -2*b*z+Bz0;


%% main loops
% 
maxstep=1000000;     
ht=0.0005;


V=1/2*(x.^2+y.^2+(gamma_z*z).^2);                         % potential in real space
D=1/2*(wx.^2+wy.^2+wz.^2);                      % potential in momentum space

V=gpuArray(single(V));
D=gpuArray(single(D));
phi1=gpuArray(single(phi1));
phi2=gpuArray(single(phi2));
phi3=gpuArray(single(phi3));
Bx=gpuArray(single(Bx));
By=gpuArray(single(By));
Bz=gpuArray(single(Bz));


    
%% use Runge-Kutta method of four order
nSx =@(phi1,phi2,phi3) real(conj(phi1).*phi2+conj(phi2).*(phi1+phi3)+conj(phi3).*phi2)/sqrt(2);
nSy =@(phi1,phi2,phi3) imag(-conj(phi1).*phi2+conj(phi2).*(phi1-phi3)+conj(phi3).*phi2)/sqrt(2);
nSz =@(phi1,phi2,phi3) real(conj(phi1).*phi1-conj(phi3).*phi3);


f1=@(phi1,phi2,phi3) -(V+beta0*(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)).*phi1+...
    -(q1*Bz+beta2*nSz(phi1,phi2,phi3)+q2/2*(2*Bz.^2+Bx.^2+By.^2)).*phi1+...
    -(q1/sqrt(2)*(Bx-1i*By)+beta2/sqrt(2)*(nSx(phi1,phi2,phi3)-1i*nSy(phi1,phi2,phi3))+q2/2*sqrt(2)*Bz.*(Bx-1i*By)).*phi2+...
    -(0+q2/2*(Bx-1i*By).^2).*phi3;


f2=@(phi1,phi2,phi3) -(V+beta0*(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)).*phi2+...
    -(q1/sqrt(2)*(Bx+1i*By)+beta2/sqrt(2)*(nSx(phi1,phi2,phi3)+1i*nSy(phi1,phi2,phi3))+q2/2*sqrt(2)*Bz.*(Bx+1i*By)).*phi1+...
    -(0+q2/2*sqrt(2)*(Bx.^2+By.^2)).*phi2+...
    -(q1/sqrt(2)*(Bx-1i*By)+beta2/sqrt(2)*(nSx(phi1,phi2,phi3)-1i*nSy(phi1,phi2,phi3))-q2/2*sqrt(2)*Bz.*(Bx-1i*By)).*phi3;


f3=@(phi1,phi2,phi3) -(V+beta0*(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)).*phi3+...
    -(0+q2/2*(Bx+1i*By).^2).*phi1+...
    -(q1/sqrt(2)*(Bx+1i*By)+beta2/sqrt(2)*(nSx(phi1,phi2,phi3)+1i*nSy(phi1,phi2,phi3))-q2/2*sqrt(2)*Bz.*(Bx+1i*By)).*phi2+...
    -(-q1*Bz-beta2*nSz(phi1,phi2,phi3)+q2/2*(2*Bz.^2+Bx.^2+By.^2)).*phi3;

tic
for nstep=1:maxstep
    nstep

    phi1temp=phi1;
    phi2temp=phi2;
    phi3temp=phi3;
    
    % solve the part of the equation in momentum space (use FFT)
    phi1 = ifftn(exp(-D*ht/2).*fftn(phi1));
    phi2 = ifftn(exp(-D*ht/2).*fftn(phi2));
    phi3 = ifftn(exp(-D*ht/2).*fftn(phi3));
    
    % use Runge-Kutta method of four order
    K11 = f1(phi1,phi2,phi3);
    K12 = f2(phi1,phi2,phi3);
    K13 = f3(phi1,phi2,phi3);
    
    K21 = f1(phi1+ht*K11/2, phi2+ht*K12/2, phi3+ht*K13/2);
    K22 = f2(phi1+ht*K11/2, phi2+ht*K12/2, phi3+ht*K13/2);
    K23 = f3(phi1+ht*K11/2, phi2+ht*K12/2, phi3+ht*K13/2);
    
    K31 = f1(phi1+ht*K21/2, phi2+ht*K22/2, phi3+ht*K23/2);
    K32 = f2(phi1+ht*K21/2, phi2+ht*K22/2, phi3+ht*K23/2);
    K33 = f3(phi1+ht*K21/2, phi2+ht*K22/2, phi3+ht*K23/2);
    
    K41 = f1(phi1+ht*K31, phi2+ht*K32, phi3+ht*K33);
    K42 = f2(phi1+ht*K31, phi2+ht*K32, phi3+ht*K33);
    K43 = f3(phi1+ht*K31, phi2+ht*K32, phi3+ht*K33);   
    
    phi1 = phi1+ht*(K11+2*K21+2*K31+K41)/6;
    phi2 = phi2+ht*(K12+2*K22+2*K32+K42)/6;
    phi3 = phi3+ht*(K13+2*K23+2*K33+K43)/6;
    
    % again FFT
    phi1 = ifftn(exp(-D*ht/2).*fftn(phi1));
    phi2 = ifftn(exp(-D*ht/2).*fftn(phi2));
    phi3 = ifftn(exp(-D*ht/2).*fftn(phi3));    
    
    % 
    P = h^3*sum(sum(sum(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)));
    phi1=phi1/sqrt(P);
    phi2=phi2/sqrt(P);
    phi3=phi3/sqrt(P);

    % establish the error
    epsilon = max(max(max(abs(phi1-phi1temp)+abs(phi2-phi2temp)+abs(phi3-phi3temp))))/3
    if (epsilon<1e-6)
        break
    end
    
end
toc

phi1=gather(phi1);
phi2=gather(phi2);
phi3=gather(phi3);

%% plot
colormap('gray')
fn=n/2;
density=max(max(max(abs(phi1).^2+abs(phi2).^2+abs(phi3).^2)));


subplot(2,3,1)
imshow(reshape(abs(phi1(:,:,fn)).^2/density,n,n));
colorbar('Position',...
    [0.93 0.25 0.030 0.53]);
xlabel('\psi_1','fontsize',24)

subplot(2,3,2)
imshow(reshape(abs(phi2(:,:,fn)).^2/density,n,n));
xlabel('\psi_2','fontsize',24)

subplot(2,3,3)
imshow(reshape(abs(phi3(:,:,fn)).^2/density,n,n));
xlabel('\psi_3','fontsize',24)

subplot(2,3,4)
imshow(reshape(abs(reshape(phi1(:,fn,:),n,n)').^2/density,n,n));
colorbar('Position',...
    [0.93 0.25 0.030 0.53]);
xlabel('\psi_1','fontsize',24)

subplot(2,3,5)
imshow(reshape(abs(reshape(phi2(:,fn,:),n,n)').^2/density,n,n));
xlabel('\psi_2','fontsize',24)

subplot(2,3,6)
imshow(reshape(abs(reshape(phi3(:,fn,:),n,n)').^2/density,n,n));
xlabel('\psi_3','fontsize',24)

set(gcf, 'position', [0 0 1200   700]);

% save data
if exist('data','dir')==0
   mkdir('data');
end
save data/GS_phi phi1 phi2 phi3
