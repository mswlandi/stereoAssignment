import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import cv2

def SSD(pixel1, pixel2):
    return (int(pixel1[0]) - int(pixel2[0]))**2 + (int(pixel1[1]) - int(pixel2[1]))**2 + (int(pixel1[2]) - int(pixel2[2]))**2

def rhoDistance(pixel1, pixel2, rho):
    D = SSD(pixel1, pixel2)
    return D / (D + rho)

def regionSSD(img1, img2, N, y, x1, x2):
    if (N % 2 == 0):
        raise Exception("N has to be odd")

    error = 0

    if (N==1):
        return SSD(img1[y, x1], img2[y, x2])

    h = in1.shape[0]
    w = in1.shape[1]

    for i in range(-(N-1)//2, +(N-1)//2):
        for j in range(-(N-1)//2, (N-1)//2):
            if (y+i >= 0 and y+i < h and
                x1+j >= 0 and x1+j < w and
                x2+j >= 0 and x2+j < w):
                error += SSD(img1[y+i, x1+j], img2[y+i, x2+j])


    # On the borders the box gets cut into a rectangle or smaller box
    actual_nx = N - max((max(x1, x2) + (N-1)//2) - w, 0) + min(min(x1, x2) - (N-1)//2, 0)
    actual_ny = N - max((y + (N-1)//2) - h, 0) + min(y - (N-1)//2, 0)
    
    # average so we don't have to change the first min_cost
    return error//(actual_nx*actual_ny)


def disparityMap(in1, in2, N, disparity_max=63):
    h = in1.shape[0]
    w = in1.shape[1]
    disparity = in1.copy()

    for y in range(0, h):
        for x1 in range(0, w):
            min_cost = 200000
            min_cost_pos = 0 # posição x do pixel de menor custo na segunda imagem

            for x2 in range(max(0, x1 - disparity_max), min(w, x1 + disparity_max)):
                current_cost = regionSSD(in1, in2, N, y, x1, x2)
                if (current_cost < min_cost):
                    min_cost = current_cost
                    min_cost_pos = x2
            
            disp = abs(min_cost_pos - x1)
            disp_scaled = disp/disparity_max * 255
            disparity[y, x1] = [disp_scaled, disp_scaled, disp_scaled]
    
    return disparity

in1 = cv2.imread('im2.png')
in2 = cv2.imread('im6.png')
disp1 = cv2.imread('disp2.png')
disp2 = cv2.imread('disp6.png')

# fig = plt.figure(figsize=(15, 20), dpi=80)
# ax = fig.add_subplot(3, 2, 1)
# ax.title.set_text('Original')
# ax.axis('off')
# plt.imshow(in1[..., ::-1])

# ax = fig.add_subplot(3, 2, 2)
# ax.title.set_text('Ground Truth')
# ax.axis('off')
# plt.imshow(disp1[...,::-1])

disparity11 = disparityMap(in1, in2, 1)
plt.imsave('dispSSD1x1.png', disparity11)

# ax = fig.add_subplot(3, 2, 3)
# ax.title.set_text('Neighbourhood = 1x1')
# ax.axis('off')
# plt.imshow(disparity11[...,::-1])

# disparity33 = disparityMap(in1, in2, 3)
# plt.imsave('dispSSD3x3.png', disparity33)

# ax = fig.add_subplot(3, 2, 4)
# ax.title.set_text('Neighbourhood = 3x3')
# ax.axis('off')
# plt.imshow(disparity33[..., ::-1])

# disparity55 = disparityMap(in1, in2, 5)
# plt.imsave('dispSSD5x5.png', disparity55)

# ax = fig.add_subplot(3, 2, 5)
# ax.title.set_text('Neighbourhood = 5x5')
# ax.axis('off')
# plt.imshow(disparity5[..., ::-1])

# disparity1111 = disparityMap(in1, in2, 11)
# plt.imsave('dispSSD11x11.png', disparity1111)

# ax = fig.add_subplot(3, 2, 6)
# ax.title.set_text('Neighbourhood = 11x11')
# ax.axis('off')
# plt.imshow(disparity1111[..., ::-1])

# fig.tight_layout(h_pad=1.5, w_pad=1.0)
# fig.savefig('dispSSDfull.png')
# plt.show()
