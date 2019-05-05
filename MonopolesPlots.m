load data/phis_40ms.mat

n=128;
colormap('gray')
fn=n/2+1;
density=max(max(max(abs(phi1_tmp(:,:,fn)).^2+abs(phi2_tmp(:,:,fn)).^2+abs(phi3_tmp(:,:,fn)).^2)));


subplot(2,3,1)
imshow(reshape(abs(phi1_tmp(:,:,fn)).^2/density,n,n));
colorbar('Position',...
    [0.93 0.25 0.030 0.53]);
xlabel('\psi_1','fontsize',24)

subplot(2,3,2)
imshow(reshape(abs(phi2_tmp(:,:,fn)).^2/density,n,n));
xlabel('\psi_0','fontsize',24)

subplot(2,3,3)
imshow(reshape(abs(phi3_tmp(:,:,fn)).^2/density,n,n));
xlabel('\psi_{-1}','fontsize',24)

subplot(2,3,4)
imshow(flipud(reshape(abs(reshape(phi1_tmp(:,fn,:),n,n)').^2/density,n,n)));
colorbar('Position',...
    [0.93 0.25 0.030 0.53]);
xlabel('\psi_1','fontsize',24)

subplot(2,3,5)
imshow(flipud(reshape(abs(reshape(phi2_tmp(:,fn,:),n,n)').^2/density,n,n)));
xlabel('\psi_0','fontsize',24)

subplot(2,3,6)
imshow(flipud(reshape(abs(reshape(phi3_tmp(:,fn,:),n,n)').^2/density,n,n)));
xlabel('\psi_{-1}','fontsize',24)

set(gcf, 'position', [0 0 1200   700]);