#!/usr/bin/env python3
"""
GalSim版本数据收集脚本
用于收集指定版本(CPP/CUDA)的测试数据

功能:
- 生成不同光子数(10K, 100K, 1M)的星系图像
- 将图像数据保存为pkl文件(用于程序化分析)
- 将图像数据保存为PNG文件(用于人工直观查看)
- 记录运行时间和统计信息

输出文件:
- {version}_photons_{n}.pkl - 包含完整测试数据
- {version}_photons_{n}.png - 可视化图像文件
- {version}_collection_summary.pkl - 汇总信息

使用方法:
    python collect_version_data.py <CPP|CUDA>
"""

import sys
import os
import time
import numpy as np
import pickle
from datetime import datetime

try:
    import galsim
    from scipy import stats
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已安装galsim、scipy和matplotlib")
    print("安装命令: pip install galsim scipy matplotlib")
    sys.exit(1)


def create_test_galaxy():
    """创建测试用的星系对象"""
    print("创建测试星系对象...")

    # 定义Sersic星系参数
    sersic_index = 4
    half_light_radius = 1.0
    flux = 1e7

    # 创建Sersic星系
    sersic_galaxy = galsim.Sersic(n=sersic_index, half_light_radius=half_light_radius, flux=flux)

    # 应用shear变换
    shear_g1 = 0.1
    shear_g2 = 0.2
    galaxy = sersic_galaxy.shear(g1=shear_g1, g2=shear_g2)

    return galaxy


def save_image_as_png(image_array, filename, version_name, num_photons):
    """将图像数组保存为PNG文件，便于人工查看"""
    try:
        # 使用对数缩放以便更好地可视化
        # 添加小的epsilon以避免log(0)
        epsilon = 1e-10
        log_image = np.log10(image_array + epsilon)

        # 创建图形
        plt.figure(figsize=(10, 8))

        # 显示图像 - 使用对数归一化和PowerNorm以获得更好的视觉效果
        im = plt.imshow(log_image, cmap='hot', interpolation='nearest',
                       norm=colors.PowerNorm(gamma=0.5))

        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('log10(Flux + ε)', rotation=270, labelpad=20)

        # 设置标题和标签
        plt.title(f'{version_name} Version - {num_photons:,} Photons\nGalSim Simulation',
                 fontsize=14, fontweight='bold')
        plt.xlabel('X (pixels)', fontsize=12)
        plt.ylabel('Y (pixels)', fontsize=12)

        # 添加网格
        plt.grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存

        print(f"   图像已保存为: {filename}")
        return True

    except Exception as e:
        print(f"⚠️ PNG保存失败: {e}")
        return False


def warmup_gpu(galaxy, version_name):
    """GPU预热运行，不保存结果"""
    if version_name.upper() == 'CUDA':
        print("\n--- GPU 预热 ---")
        try:
            # 使用1个光子进行预热
            image_size = 256
            pixel_scale = 0.2
            image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
            rng = galsim.UniformDeviate(22222)

            # 预热运行
            galaxy.drawImage(image=image, method='phot', n_photons=1, rng=rng)
            print("✅ GPU预热完成")
        except Exception as e:
            print(f"⚠️ GPU预热失败: {e}")


def generate_and_save_data(version_name):
    """生成并保存指定版本的数据"""
    print(f"\n=== {version_name}版本数据收集开始 ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    galaxy = create_test_galaxy()

    # GPU预热（仅CUDA版本需要）
    # warmup_gpu(galaxy, version_name)

    # 测试规模：包含1个预热项和4个实际测试项
    num_photons_list = [1, 10000, 100000, 1000000]


    # 确保results目录存在
    os.makedirs('results', exist_ok=True)

    summary_data = {
        'version': version_name,
        'collection_time': datetime.now().isoformat(),
        'test_cases': [],
        'success': False
    }

    success_count = 0

    for i, num_photons in enumerate(num_photons_list):
        try:
            # 创建图像对象
            image_size = 256
            pixel_scale = 0.2
            image = galsim.ImageF(image_size, image_size, scale=pixel_scale)

            # 使用固定随机种子
            rng = galsim.UniformDeviate(22222)

            # 计时开始
            start_time = time.time()

            # 生成图像
            galaxy.drawImage(image=image, method='phot', n_photons=num_photons, rng=rng)

            # 计时结束
            end_time = time.time()
            runtime = (end_time - start_time) * 1000  # 毫秒
            image_array = image.array

            
            if i > 0:
                print(f"\n--- {num_photons:,} 光子测试 ---")                
                # 计算统计信息
                stats_data = {
                    'mean': float(np.mean(image_array)),
                    'std': float(np.std(image_array)),
                    'min': float(np.min(image_array)),
                    'max': float(np.max(image_array)),
                    'sum': float(np.sum(image_array)),
                    'nonzero_count': int(np.count_nonzero(image_array)),
                    'central_pixel': float(image_array[image_array.shape[0]//2, image_array.shape[1]//2]),
                    'skewness': float(stats.skew(image_array.flatten())),
                    'kurtosis': float(stats.kurtosis(image_array.flatten()))
                }

                # 准备保存的数据
                test_result = {
                    'version': version_name,
                    'num_photons': num_photons,
                    'image': image_array,
                    'statistics': stats_data,
                    'runtime_ms': runtime,
                    'timestamp': time.time(),
                    'success': True
                }

                # 保存数据
                pkl_filename = f"results/{version_name.lower()}_photons_{num_photons}.pkl"
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(test_result, f)

                # 保存图像为PNG文件
                png_filename = f"results/{version_name.lower()}_photons_{num_photons}.png"
                png_success = save_image_as_png(image_array, png_filename, version_name, num_photons)

                print(f"✅ 运行时间: {runtime:.2f} 毫秒")
                print(f"   总通量: {stats_data['sum']:.2f}")
                print(f"   非零像素数: {stats_data['nonzero_count']}")
                print(f"   数据已保存到: {pkl_filename}")

                # 记录测试用例信息
                summary_data['test_cases'].append({
                    'num_photons': num_photons,
                    'runtime_ms': runtime,
                    'flux': stats_data['sum'],
                    'nonzero_count': stats_data['nonzero_count'],
                    'pkl_filename': pkl_filename,
                    'png_filename': png_filename,
                    'png_saved': png_success,
                    'success': True
                })

                success_count += 1

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            # 记录失败的测试用例
            summary_data['test_cases'].append({
                'num_photons': num_photons,
                'error': str(e),
                'success': False
            })


    # 保存总结数据
    summary_data['success'] = success_count == len(num_photons_list) - 1  # 除去预热项
    summary_data['success_count'] = success_count
    summary_data['total_tests'] = len(num_photons_list) - 1  # 除去预热项

    summary_filename = f"results/{version_name.lower()}_collection_summary.pkl"
    with open(summary_filename, 'wb') as f:
        pickle.dump(summary_data, f)

    print(f"\n=== {version_name}版本数据收集完成 ===")
    print(f"成功测试: {success_count}/{len(num_photons_list)-1}")
    print(f"总结文件: {summary_filename}")

    # 显示生成的PNG文件
    png_files = [case['png_filename'] for case in summary_data['test_cases'] if case.get('png_saved')]
    if png_files:
        print(f"\n📸 生成的PNG图像文件:")
        for png_file in png_files:
            print(f"   • {png_file}")
        print("   这些图像可以直接查看以直观比较不同版本的输出效果")

    return summary_data['success']


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python collect_version_data.py <CPP|CUDA>")
        sys.exit(1)

    version_name = sys.argv[1].upper()
    if version_name not in ['CPP', 'CUDA']:
        print("错误: 版本必须是 CPP 或 CUDA")
        sys.exit(1)

    print("GalSim版本数据收集工具")
    print(f"目标版本: {version_name}")
    print(f"GalSim版本: {galsim.__version__}")
    print(f"Python版本: {sys.version}")

    success = generate_and_save_data(version_name)

    if success:
        print("\n🎉 数据收集成功完成!")
        sys.exit(0)
    else:
        print("\n❌ 数据收集过程中出现错误")
        sys.exit(1)


if __name__ == "__main__":
    main()