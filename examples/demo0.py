import galsim
import time
# 定义Sersic星系参数
sersic_index = 4  # Sersic指数
half_light_radius = 1.0  # 半光半径（单位：弧秒）
flux = 1e7  # 总光通量

# 创建Sersic星系对象
sersic_galaxy1 = galsim.Sersic(n=sersic_index, half_light_radius=half_light_radius, flux=flux)
sersic_galaxy2 = galsim.Sersic(n=sersic_index+1, half_light_radius=half_light_radius, flux=flux)
sersic_galaxy3 = galsim.Sersic(n=sersic_index+2, half_light_radius=half_light_radius, flux=flux)

# 定义光子射击参数````````     
num_photons = 1e8  # 光子数量

# 创建图像对象
image_size = 256  # 图像大小（像素）                   
pixel_scale = 0.2  # 像素比例（弧秒/像素）
image = galsim.ImageF(image_size, image_size, scale=pixel_scale)

# 进行光子射击渲染
# 统计耗时

rng = galsim.UniformDeviate(22222)  # 随机数生成器
start = time.time()
sersic_galaxy1.drawImage(image=image, method='phot', n_photons=num_photons, rng=rng)
end = time.time()
# 打印毫秒
print('all time : {:.0f} ms'.format((end-start) * 1000))
print('-------')
start = time.time()
sersic_galaxy2.drawImage(image=image, method='phot', n_photons=num_photons, rng=rng)
end = time.time()
# 打印毫秒
print('all time : {:.0f} ms'.format((end-start) * 1000))
print('-------')
start = time.time()
sersic_galaxy3.drawImage(image=image, method='phot', n_photons=num_photons, rng=rng)
end = time.time()
# 打印毫秒
print('all time : {:.0f} ms'.format((end-start) * 1000))
print('-------')

# 保存图像
image.write('sersic_galaxy.fits')

print("Sersic星系仿真图像已保存为'sersic_galaxy.fits'")