from pytorch_msssim import ssim

class SsimLoss:
    def __call__(self, a, b):
        return 1 - ssim(a, b, data_range=1, size_average=True)
