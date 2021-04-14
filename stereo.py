import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import cv2
from math import sqrt
from scipy import ndimage

def SSD(pixel1, pixel2):
    return (int(pixel1[0]) - int(pixel2[0]))**2 + (int(pixel1[1]) - int(pixel2[1]))**2 + (int(pixel1[2]) - int(pixel2[2]))**2

def rhoDistance(pixel1, pixel2, rho):
    D = sqrt(SSD(pixel1, pixel2))
    return D / (D + rho)

def labSquaredDistance(pixel1, pixel2):
    labPixel1 = cv2.cvtColor( np.uint8([[pixel1]] ), cv2.COLOR_BGR2LAB)[0][0]
    labPixel2 = cv2.cvtColor( np.uint8([[pixel2]] ), cv2.COLOR_BGR2LAB)[0][0]
    return SSD(labPixel1, labPixel2)

def regionFunction(img1, img2, N, y, x1, x2, distanceType="SSD"):
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
    return error/(actual_nx*actual_ny)


def disparityMap(in1, in2, N, disparity_max=63, distanceType="SSD"):
    h = in1.shape[0]
    w = in1.shape[1]
    disparity = in1.copy()

    if(distanceType == "LAB"):
        img1 = cv2.cvtColor(in1, cv2.COLOR_BGR2LAB)
        img2 = cv2.cvtColor(in2, cv2.COLOR_BGR2LAB)
    else:
        img1 = in1
        img2 = in2

    for y in range(0, h):
        for x1 in range(0, w):
            min_cost = 900000
            min_cost_pos = 0 # posição x do pixel de menor custo na segunda imagem

            for x2 in range(max(0, x1 - disparity_max), min(w, x1 + disparity_max)):
                current_cost = regionFunction(img1, img2, N, y, x1, x2, distanceType)

                if (current_cost < min_cost):
                    min_cost = current_cost
                    min_cost_pos = x2
            
            disp = abs(min_cost_pos - x1)
            disp_scaled = disp/disparity_max * 255
            disparity[y, x1] = [disp_scaled, disp_scaled, disp_scaled]
    
    return disparity


def disparityMapAggregation(in1, in2, N, M, disparity_max=63, distanceType="SSD", aggregationType="median"):
    h = in1.shape[0]
    w = in1.shape[1]

    if(distanceType == "LAB"):
        img1 = cv2.cvtColor(in1, cv2.COLOR_BGR2LAB)
        img2 = cv2.cvtColor(in2, cv2.COLOR_BGR2LAB)

    # "volume" de custos, lista de imagens de custo, onde o índice é uma disparidade específica
    costVolume = []
    for d in range(disparity_max):
        costs = np.zeros((h,w))

        for y in range(0, h):
            for x1 in range(0, w):
                cost = 0

                # para cara disparidade, pega o menor custo, dentre a disparidade d para a esquerda e para a direita
                if (x1 - d < 0):
                    cost = regionFunction(in1, in2, N, y, x1, x1 + d, distanceType)
                elif (x1 + d >= w):
                    cost = regionFunction(in1, in2, N, y, x1, x1 - d, distanceType)
                else:
                    cost = min(regionFunction(in1, in2, N, y, x1, x1 - d, distanceType),
                            regionFunction(in1, in2, N, y, x1, x1 + d, distanceType))
                
                # coloca o custo na imagem de custos
                costs[y, x1] = cost
        
        # agregação em si
        if (aggregationType == "median"):
            costs = ndimage.median_filter(costs, size=M)
        else:
            costs = ndimage.uniform_filter(costs, size=M)

        # coloca na lista de imagens de custos
        costVolume.append(costs)
    

    # cria a imagem com disparidades de menor custo
    disparityImage = in1.copy()
    for y in range(0,h):
        for x in range(0,w):
            min_cost = costVolume[0][y, x]
            disparity = 0 # disparidade com menor custo para o (x,y) pixel, dentre as imagens de custo por disparidade
            for i in range(len(costVolume)):
                if (costVolume[i][y, x] < min_cost):
                    min_cost = costVolume[i][y, x]
                    disparity = i
            disp_scaled = disparity/disparity_max * 255
            disparityImage[y, x] = [disp_scaled, disp_scaled, disp_scaled]
    
    return disparityImage


def appendImages(img1, img2):
    return np.append(np.append(img1, np.full((img1.shape[0], 15, 3), np.uint8(255)), axis=1), img2, axis=1)



# Gera imagens para a primeira parte (a) do exercício 1
in1 = cv2.imread('im2.png')
in2 = cv2.imread('im6.png')
disp1 = cv2.imread('disp2.png')
disp2 = cv2.imread('disp6.png')

fig = plt.figure(figsize=(15, 20), dpi=80)
ax = fig.add_subplot(3, 2, 1)
ax.title.set_text('Original')
ax.axis('off')
plt.imshow(in1[..., ::-1])

ax = fig.add_subplot(3, 2, 2)
ax.title.set_text('Ground Truth')
ax.axis('off')
plt.imshow(disp1[...,::-1])

disparity11 = disparityMap(in1, in2, 1)
plt.imsave('dispSSD1x1.png', disparity11)

ax = fig.add_subplot(3, 2, 3)
ax.title.set_text('Neighbourhood = 1x1')
ax.axis('off')
plt.imshow(disparity11[...,::-1])

disparity33 = disparityMap(in1, in2, 3)
plt.imsave('dispSSD3x3.png', disparity33)

ax = fig.add_subplot(3, 2, 4)
ax.title.set_text('Neighbourhood = 3x3')
ax.axis('off')
plt.imshow(disparity33[..., ::-1])

disparity55 = disparityMap(in1, in2, 5)
plt.imsave('dispSSD5x5.png', disparity55)

ax = fig.add_subplot(3, 2, 5)
ax.title.set_text('Neighbourhood = 5x5')
ax.axis('off')
plt.imshow(disparity55[..., ::-1])

disparity1111 = disparityMap(in1, in2, 11)
plt.imsave('dispSSD11x11.png', disparity1111)

ax = fig.add_subplot(3, 2, 6)
ax.title.set_text('Neighbourhood = 11x11')
ax.axis('off')
plt.imshow(disparity1111[..., ::-1])

fig.tight_layout(h_pad=1.5, w_pad=1.0)
fig.savefig('dispSSDfull.png')


# Gera imagens para a segunda parte (b) do exercício 1
in1 = cv2.imread('im2.png')
in2 = cv2.imread('im6.png')

disparityLAB11 = disparityMap(in1, in2, 1, distanceType="LAB")
plt.imsave('dispSSD1x1_LAB.png', disparityLAB11)
plt.imsave('dispSSD1x1_LAB_Comparison.png', appendImages(disparity11, disparityLAB11))

disparityLAB33 = disparityMap(in1, in2, 3, distanceType="LAB")
plt.imsave('dispSSD3x3_LAB.png', disparityLAB33)
plt.imsave('dispSSD3x3_LAB_Comparison.png', appendImages(disparity33, disparityLAB33))

disparityLAB55 = disparityMap(in1, in2, 5, distanceType="LAB")
plt.imsave('dispSSD5x5_LAB.png', disparityLAB55)
plt.imsave('dispSSD5x5_LAB_Comparison.png', appendImages(disparity55, disparityLAB55))

disparityLAB1111 = disparityMap(in1, in2, 11, distanceType="LAB")
plt.imsave('dispSSD11x11_LAB.png', disparityLAB1111)
plt.imsave('dispSSD11x11_LAB_Comparison.png', appendImages(disparity1111, disparityLAB1111))

# Gera imagens para o exercício 2
# Imagens com filtro de média
disparityMeanAgg11 = disparityMapAggregation(in1, in2, 1, 3, distanceType="SSD", aggregationType="mean")
plt.imsave('dispMeanAgg1x1_Agg.png', disparityMeanAgg11)
plt.imsave('dispMeanAgg1x1_Agg_Comparison.png', appendImages(disparity11, disparityMeanAgg11))

disparityMeanAgg33 = disparityMapAggregation(in1, in2, 3, 3, distanceType="SSD", aggregationType="mean")
plt.imsave('dispMeanAgg3x3_Agg.png', disparityMeanAgg33)
plt.imsave('dispMeanAgg3x3_Agg_Comparison.png', appendImages(disparity33, disparityMeanAgg33))

disparityMeanAgg55 = disparityMapAggregation(in1, in2, 5, 3, distanceType="SSD", aggregationType="mean")
plt.imsave('dispMeanAgg5x5_Agg.png', disparityMeanAgg55)
plt.imsave('dispMeanAgg5x5_Agg_Comparison.png', appendImages(disparity55, disparityMeanAgg55))

disparityMeanAgg1111 = disparityMapAggregation(in1, in2, 11, 3, distanceType="SSD", aggregationType="mean")
plt.imsave('dispMeanAgg11x11_Agg.png', disparityMeanAgg1111)
plt.imsave('dispMeanAgg11x11_Agg_Comparison.png', appendImages(disparity1111, disparityMeanAgg1111))

# Imagens com filtro de mediana
disparityMedianAgg11 = disparityMapAggregation(in1, in2, 1, 3, distanceType="SSD", aggregationType="median")
plt.imsave('dispMedianAgg1x1_Agg.png', disparityMedianAgg11)
plt.imsave('dispMedianAgg1x1_Agg_Comparison.png', appendImages(disparity11, disparityMedianAgg11))

disparityMedianAgg33 = disparityMapAggregation(in1, in2, 3, 3, distanceType="SSD", aggregationType="median")
plt.imsave('dispMedianAgg3x3_Agg.png', disparityMedianAgg33)
plt.imsave('dispMedianAgg3x3_Agg_Comparison.png', appendImages(disparity33, disparityMedianAgg33))

disparityMedianAgg55 = disparityMapAggregation(in1, in2, 5, 3, distanceType="SSD", aggregationType="median")
plt.imsave('dispMedianAgg5x5_Agg.png', disparityMedianAgg55)
plt.imsave('dispMedianAgg5x5_Agg_Comparison.png', appendImages(disparity55, disparityMedianAgg55))

disparityMedianAgg1111 = disparityMapAggregation(in1, in2, 11, 3, distanceType="SSD", aggregationType="median")
plt.imsave('dispMedianAgg11x11_Agg.png', disparityMedianAgg1111)
plt.imsave('dispMedianAgg11x11_Agg_Comparison.png', appendImages(disparity1111, disparityMedianAgg1111))

# Imagens de média, N=3, M variando
disparityMeanAggM3 = disparityMapAggregation(in1, in2, 3, 3, distanceType="SSD", aggregationType="mean")
plt.imsave('dispMeanAggM3.png', disparityMeanAggM3)

disparityMeanAggM5 = disparityMapAggregation(in1, in2, 3, 5, distanceType="SSD", aggregationType="mean")
plt.imsave('dispMeanAggM5.png', disparityMeanAggM5)

disparityMeanAggM7 = disparityMapAggregation(in1, in2, 3, 7, distanceType="SSD", aggregationType="mean")
plt.imsave('dispMeanAggM7.png', disparityMeanAggM7)

disparityMeanAggM11 = disparityMapAggregation(in1, in2, 3, 11, distanceType="SSD", aggregationType="mean")
plt.imsave('dispMeanAggM11.png', disparityMeanAggM11)

# Imagens de mediana, N=3, M variando
disparityMedianAggM3 = disparityMapAggregation(in1, in2, 3, 3, distanceType="SSD", aggregationType="median")
plt.imsave('dispMedianAggM3.png', disparityMedianAggM3)

disparityMedianAggM5 = disparityMapAggregation(in1, in2, 3, 5, distanceType="SSD", aggregationType="median")
plt.imsave('dispMedianAggM5.png', disparityMedianAggM5)

disparityMedianAggM7 = disparityMapAggregation(in1, in2, 3, 7, distanceType="SSD", aggregationType="median")
plt.imsave('dispMedianAggM7.png', disparityMedianAggM7)

disparityMedianAggM11 = disparityMapAggregation(in1, in2, 3, 11, distanceType="SSD", aggregationType="median")
plt.imsave('dispMedianAggM11.png', disparityMedianAggM11)
