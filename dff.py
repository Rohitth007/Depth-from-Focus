import os
import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from focus_measure import (
    sum_modified_laplacian,
    estimate_depth_per_pixel,
    imshow,
    imgraysc,
)


####### DATASET EXTRACTION #######################################################
stack_number = input(
    "\nWhich focus stack do you want to find the depth map of?\n(Type 'custom' for custom folder)\n> "
)

if stack_number != "custom":
    focus_stack_path = "/home/rohitth007/Documents/MOOCs/Computer Vision/Prof.Rajagopalan Labs/Depth From Focus/dataset/focus_stack"
    focus_stack = sorted(
        [
            img
            for img in os.listdir(focus_stack_path)
            if img.startswith(stack_number + "__")
        ]
    )
else:
    focus_stack_path = input("Input path to folder > ")
    focus_stack = sorted([img for img in os.listdir(focus_stack_path)])

display_raw_images = input("\nDisplay original stack images? [y/n] > ")
for i, img_name in enumerate(focus_stack):
    img = cv.imread(focus_stack_path + "/" + img_name, cv.IMREAD_COLOR)
    focus_stack[i] = img
    if display_raw_images == "y":
        imshow(img_name, img)
    else:
        print(f"Importing {img_name}...")
cv.destroyAllWindows()


####### DEPTH ESTIMATION #############################################
display_sml = input("\nDisplay fxx, fyy, ML, SML images? [y/n] > ")
print("Using Sum Modified Laplacian as Focus Measure...")
sml_focus_stack = sum_modified_laplacian(focus_stack, display_sml)

# Gives depth in terms of stack units
print("Estimating Depth using Gaussian Interpolation...")
depth_map = estimate_depth_per_pixel(sml_focus_stack)


####### ALL-FOCUSED IMAGE ############################################
def reconstruct_all_focus_image(focus_stack, depth_map):
    def linear_interpolation(y_1, y_2, x_1, x):
        """Special case x_2 - x_1 = 1"""
        y_1 = int(y_1)
        y_2 = int(y_2)
        a = y_2 - y_1
        b = y_1 - a * x_1
        return a * x + b

    h, w = depth_map.shape
    all_focus = np.zeros((h, w, 3))
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth = min(depth_map[i, j], len(focus_stack) - 1)
            d_floor = math.floor(depth)
            d_ceil = math.ceil(depth)

            for k in range(3):
                i_ceil = focus_stack[d_ceil][i, j, k]
                i_floor = focus_stack[d_floor][i, j, k]
                i_depth = linear_interpolation(i_floor, i_ceil, d_floor, depth)
                all_focus[i, j, k] = i_depth

    return imgraysc(all_focus)


print("\nReconstructing All Focused Image...")
all_focus = reconstruct_all_focus_image(focus_stack, depth_map)
imshow("First Image in Stack", focus_stack[0])
imshow("Last Image in Stack", focus_stack[-1])
imshow("Reconstructed All-Focus Image\n", all_focus)


####### VISUALIZATION ################################################
def depth_plot_2D(depth_map):
    denoised_depth_map = cv.medianBlur(np.array(depth_map, np.uint8), 3)
    imshow("Depth Map 2D - Median Blurred", denoised_depth_map)


def depth_plot_3D(depth_map):
    print(":::::Displaying Depth Plot 3D\n")
    denoised_depth_map = cv.medianBlur(np.array(depth_map, np.uint8), 5)

    def f(x, y):
        return 255 - denoised_depth_map[x, y]

    x = range(depth_map.shape[0])
    y = range(depth_map.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="green")
    plt.show()


depth_map = imgraysc(depth_map)
depth_plot_2D(depth_map)
depth_plot_3D(depth_map)
