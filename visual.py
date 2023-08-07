import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载深度图和视差图
depth_map = cv2.imread('output/gray/000000_10.png', cv2.IMREAD_GRAYSCALE)
disparity_map = cv2.imread('test_images/kitti/disp/000000_10.png', cv2.IMREAD_GRAYSCALE)

# 创建GT图的二值掩码
mask = np.where(disparity_map > 0, 255, 0).astype(np.uint8)


# 归一化数据范围
depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
disparity_map_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)

# 计算深度图和视差图的差异
diff_map = cv2.absdiff(depth_map_norm, disparity_map_norm)
diff_map = cv2.bitwise_and(diff_map, mask)

# 可视化差异图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(depth_map_norm, cmap='gray')
plt.title('Depth Map')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(disparity_map_norm, cmap='gray')
plt.title('Disparity Map')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(diff_map, cmap='jet')
plt.title('Difference Map')
plt.colorbar()
plt.axis('off')

diff_map1 = cv2.cvt
cv2.imwrite("diff_map.png", diff_map)

plt.tight_layout()
plt.show()