import numpy as np
import matplotlib.pyplot as plt

def plot_psnr_ssim(psnr, ssim, loss, epoch, filename):
    x1 = np.linspace(1, epoch, epoch)
    x2 = np.linspace(1, epoch, epoch)
    x3 = np.linspace(1, epoch, epoch)

    y1 = psnr
    y2 = ssim
    y3 = loss

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('psnr/ssim')

    ax1.plot(x1, y1, 'o-')
    ax1.set_ylabel('psnr')

    ax2.plot(x2, y2, '.-')
    ax2.set_ylabel('ssim')

    ax3.plot(x3, y3, 'v-')
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('loss')

    fig.savefig(filename + '/psnr_ssim_loss.svg')