import galsim
import numpy as np

# 初始化参数
pixSize = 0.1  # 假设像素大小
folding_threshold = 1e-3  # 假设的GSParams参数
pos_img = galsim.PositionD(x=128, y=128)  # 假设图像中的位置
imPSF = np.random.rand(256, 256)  # 假设的PSF数据
n_photons = 1e6  # 光子数量

# 创建一个257x257的空数组，并将imPSF放入其中
imPSFt = np.zeros([257, 257])
imPSFt[0:256, 0:256] = imPSF

# 创建Galsim图像对象
print('----------------------1----------------------')
img = galsim.ImageF(imPSFt, scale=pixSize)

# 定义GSParams
print('----------------------2----------------------')
gsp = galsim.GSParams(folding_threshold=folding_threshold)

# 创建InterpolatedImage
print('----------------------3----------------------')
psf = galsim.InterpolatedImage(img, gsparams=gsp)

# 定义一个简单的WCS，这里使用PixelScale作为例子
print('----------------------4----------------------')
wcs = galsim.PixelScale(scale=pixSize)

# 将PSF图像从图像坐标转换为世界坐标
print('----------------------5----------------------')
psf = wcs.toWorld(psf, image_pos=pos_img)

# 使用光子射击进行渲染
# 创建一个空图像用于渲染PSF
print('----------------------6----------------------') 
psf_image = galsim.ImageF(257, 257, scale=pixSize)

# 执行光子射击渲染
print('----------------------7----------------------')
psf.drawImage(image=psf_image, method='phot', n_photons=n_photons)

# # 将渲染的PSF图像保存到FITS文件中
# output_filename = "psf_photon_shooting.fits"
# psf_image.write(output_filename)

# print(f"PSF rendered with photon shooting and saved to {output_filename}")
