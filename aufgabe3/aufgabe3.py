import sys, os
import numpy as np
from math import sin
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import pascal


def main():
    aufgabe1(True)
    # aufgabe2()
    # aufgabe3()
    # aufgabe4()


# ==================================
# 1 Edge Detection
# ==================================


def aufgabe1(scale=False):
    if scale:
        for scale in [0.5, 1, 2]:
            edgeDetection(scale)
        plt.show()
    else:
        edgeDetection()
        plt.show()


def edgeDetection(scale=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    # image = Image.open("./test.png")
    image = cv2.imread("lenna.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(
        image,
        (int(image.shape[0] * scale), int(image.shape[1] * scale)),
        interpolation=cv2.INTER_AREA,
    )

    laplaceEdge(image, ax1)
    sobelEdge_improved(image, ax2)
    logEdge(image, ax3)
    dogEdge(image, ax4)

    fig.suptitle("Skalierung x%s" % scale)


# Vorteile: viele feine Details, dünne Kantenlinien
# Nachteile: viel Rauschen
def laplaceEdge(image, ax):
    ax.title.set_text("Laplace")
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    ax.imshow(image, cmap="gray")


# Vorteile: sehr wenig Rauschen, nur "wichtigste Kanten"
# Nachteile: sehr unterschiedliche Kantenlinien-Stärke, häufig nicht durchgängig (Richtung der Kanten?)
def sobelEdge(image, ax):
    ax.title.set_text("Sobel")
    kernel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) * 1 / 8
    kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * 1 / 8
    image = cv2.filter2D(image, -1, kernel1)
    image = cv2.filter2D(image, -1, kernel2)
    ax.imshow(image, cmap="gray")


def sobelEdge_improved(image, ax):
    ax.title.set_text("Sobel (improved)")
    kernel1 = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]]) * 1 / 32
    kernel2 = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]) * 1 / 32
    image = cv2.filter2D(image, -1, kernel1)
    image = cv2.filter2D(image, -1, kernel2)
    ax.imshow(image, cmap="gray")


# Vorteile: sehr klar erkennbare Kanten (dicke) in jede Richtung
# Nachteile: mehr Rechenzeit
def logEdge(image, ax):
    ax.title.set_text("LoG")
    kernel = (
        np.array(
            [
                [0, 1, 2, 1, 0],
                [1, 0, -2, 0, 1],
                [2, -2, -8, -2, 2],
                [1, 0, -2, 0, 1],
                [0, 1, 2, 1, 0],
            ]
        )
        * 1
        / 16
    )
    image = cv2.filter2D(image, -1, kernel)
    ax.imshow(image, cmap="gray")


# wie LoG, nur etwas klarer, weniger Rauschen
def dogEdge(image, ax):
    ax.title.set_text("DoG")
    kernel = (
        np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 0, -8, 0, 4],
                [6, -8, -28, -8, 6],
                [4, 0, -8, 0, 4],
                [0, 4, 6, 4, 1],
            ]
        )
        * 1
        / 16
    )
    image = cv2.filter2D(image, -1, kernel)
    ax.imshow(image, cmap="gray")


def normalizeKernel(kernel):
    total = np.sum(kernel)
    return np.multiply(kernel, 1 / total)


# ==================================
# 2 Laplace Pyramid
# ==================================


def aufgabe2():
    name = "lenna"
    image = cv2.imread(f"{name}.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # test, _ = reduce(image)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # up1 = expand(test)
    # up2 = cv2.pyrUp(test)
    # # ax1.imshow(test, cmap="gray")
    # diff = cv2.subtract(up1, up2)
    # print(diff)
    # ax2.imshow(up1, cmap="gray")
    # ax3.imshow(up2, cmap="gray")
    # ax1.imshow(diff)
    # plt.show()

    lastImage, pyramid = laplacePyramid(image, name)
    reconImage = reconstructImageFromPyramid(lastImage, pyramid)
    plt.imshow(reconImage, cmap="gray")
    plt.axis("off")
    plt.show()


def laplacePyramid(image, name, steps=4):
    gaussPyramid = [image]
    lastImage = image
    cv2.imwrite(f"{name}_pyramid_0.png", image)
    for i in range(1, steps + 1):
        down, _ = reduce(lastImage)
        gaussPyramid.append(down)
        cv2.imwrite(f"{name}_pyramid_{i}.png", down)
        lastImage = down

    pyramid = [gaussPyramid[-1]]
    for i in range(steps, 0, -1):
        img = gaussPyramid[i]
        ex = expand(img)
        diff = cv2.subtract(gaussPyramid[i - 1], ex)
        pyramid.append(diff)
        cv2.imwrite(f"{name}_pyramid_laplace_{i}.png", diff)

        # plt.imshow(diff, cmap="gray")
        # plt.show()

    return lastImage, pyramid


def reconstructImageFromPyramid(lastImage, pyramid):
    steps = len(pyramid)
    for i in range(steps - 1):
        lastImage = expand(lastImage)
        lastImage = cv2.add(lastImage, pyramid[i + 1])

    return lastImage


def reduce(image):
    kernel = gaussFilter(5)
    blurred = cv2.filter2D(image, -1, kernel)
    diff = cv2.subtract(image, blurred)
    down = blurred[::2, ::2]
    return down, diff


def expand(image):
    height, width = image.shape[0] * 2, image.shape[1] * 2
    newImage = np.zeros((height, width), np.uint8)
    newImage[::2, ::2] = image
    kernel = gaussFilter(5)
    newImage = cv2.filter2D(newImage, -1, kernel)
    newImage = np.multiply(newImage, 4)
    return newImage


def gaussFilter(size):
    triangle = pascal(size, kind="lower")  # Pascal triangle
    pRow = triangle[-1]  # last row
    kernel = np.outer(pRow, pRow)  # last row * last row
    return normalizeKernel(kernel)


# ==================================
# 3 Dilatation und Erosion
# ==================================


def aufgabe3():
    image = cv2.imread("test.png")
    sizeMult = 0.2
    image = cv2.resize(
        image,
        (int(image.shape[0] * sizeMult), int(image.shape[1] * sizeMult)),
        interpolation=cv2.INTER_AREA,
    )
    image = makeImageBinary(image)

    plt.imshow(image, cmap="gray")
    plt.axis("off")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    binaryDilatation(image, ax1)
    binaryErosion(image, ax2)
    binaryOpening(image, ax3)
    binaryClosing(image, ax4)

    plt.show()


def makeImageBinary(image: cv2.Mat):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thres = 127  # just half for now
    return cv2.threshold(image, thres, 255, cv2.THRESH_BINARY)[1]


# Swapped functions??
def binaryDilatation(image, ax):
    ax.title.set_text("Dilatation")
    kernel = np.full((3, 3), 1)
    image = cv2.erode(image, kernel, iterations=1)
    ax.imshow(image, cmap="gray")


def binaryErosion(image, ax):
    ax.title.set_text("Erosion")
    kernel = np.full((3, 3), 1)
    image = cv2.dilate(image, kernel, iterations=1)
    ax.imshow(image, cmap="gray")


def binaryOpening(image, ax):
    ax.title.set_text("Opening")
    kernel = np.full((3, 3), 1)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    ax.imshow(image, cmap="gray")


def binaryClosing(image, ax):
    ax.title.set_text("Closing")
    kernel = np.full((3, 3), 1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=1)
    ax.imshow(image, cmap="gray")


# ==================================
# 4 Ausdünnen
# ==================================


def aufgabe4():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    image = cv2.imread("horse.png")
    sizeMult = 0.2
    # image = cv2.resize(
    #     image,
    #     (int(image.shape[0] * sizeMult), int(image.shape[1] * sizeMult)),
    #     interpolation=cv2.INTER_AREA,
    # )
    image = makeImageBinary(image)
    ax1.imshow(255 - image, cmap="gray")

    # image = 255 - image  # invert here
    newImage = image

    lastCount = 0
    count = 0
    index = 0
    while lastCount == 0 or count != lastCount:
        lastCount = count
        print("Step %s..." % index, end="\r")
        index += 1
        newImage = skeletonStep(newImage)
        count = np.sum(np.array(newImage))

    print("\nDone!")
    ax2.imshow(255 - newImage, cmap="gray")

    ax1.axis("off")
    ax2.axis("off")

    plt.show()


def skeletonStep(image):
    temp = image

    kernel = np.array([[-1, 0, 1], [-1, 1, 1], [-1, 0, 1]])
    im1 = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    temp = cv2.bitwise_and(255 - im1, temp)

    kernel = np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 1]])
    im2 = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    temp = cv2.bitwise_and(255 - im2, temp)

    kernel = np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]])
    im3 = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    temp = cv2.bitwise_and(255 - im3, temp)

    kernel = np.array([[0, -1, -1], [1, 1, -1], [1, 1, 0]])
    im4 = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    temp = cv2.bitwise_and(255 - im4, temp)

    kernel = np.array([[1, 0, -1], [1, 1, -1], [1, 0, -1]])
    im5 = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    temp = cv2.bitwise_and(255 - im5, temp)

    kernel = np.array([[1, 1, 0], [1, 1, -1], [0, -1, -1]])
    im6 = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    temp = cv2.bitwise_and(255 - im6, temp)

    kernel = np.array([[1, 1, 1], [0, 1, 0], [-1, -1, -1]])
    im7 = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    temp = cv2.bitwise_and(255 - im7, temp)

    kernel = np.array([[0, 1, 1], [-1, 1, 1], [-1, -1, 0]])
    im8 = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    temp = cv2.bitwise_and(255 - im8, temp)

    return temp


if __name__ == "__main__":
    main()
