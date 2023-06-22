from SIFT_algo import SIFT_Algorithm
from SIFT_KeyPoint import SIFT_KeyPoint
from SIFT_Params import SIFT_Params
from SIFT_Visualization import visualize_keypoints, visualize_scale_space

import numpy as np
import cv2
from scipy.linalg import pascal
from math import sqrt
import matplotlib.pyplot as plt


def main():
    name = "mage"
    image = cv2.imread(f"{name}.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255

    params = SIFT_Params()
    scale_space, deltas, sigmas = create_scale_space(image, params)
    # visualize_scale_space(scale_space[0], "Scale Space - Selfmade", False)
    dogs = create_dogs(scale_space)
    # visualize_scale_space(dogs, "DoGs - Selfmade", False)
    keypoints = find_discrete_extremas(dogs)
    visualize_keypoints(
        scale_space, keypoints, deltas, "Keypoints - Selfmade", show=False
    )
    # filtered_keypoints = SIFT_Algorithm.filter_extremas(keypoints, dogs, params)
    # visualize_keypoints(
    #     scale_space,
    #     filtered_keypoints,
    #     deltas,
    #     "Filtered Keypoints - Selfmade",
    #     show=False,
    # )

    # scale_space_2, deltas_2, sigmas_2 = SIFT_Algorithm.create_scale_space(image, params)
    # visualize_scale_space(scale_space_2[0], "Scale Space - Premade", False)
    # dogs_2 = SIFT_Algorithm.create_dogs(scale_space_2, params)
    # visualize_scale_space(dogs_2, "DoGs - Premade", False)
    # keypoints_2 = SIFT_Algorithm.find_discrete_extremas(
    #     dogs_2, params, sigmas_2, deltas_2
    # )
    # visualize_keypoints(
    #     scale_space, keypoints_2, deltas_2, "Keypoints - Premade", show=False
    # )
    plt.show()


def create_scale_space(
    image: np.ndarray, params: SIFT_Params
) -> tuple[list[list[np.ndarray]], list[float], list[list[float]]]:
    print("Creating Scale Space...", end="\r")
    image = cv2.resize(
        image,
        (0, 0),
        fx=1 / params.delta_min,
        fy=1 / params.delta_min,
        interpolation=cv2.INTER_LINEAR,
    )

    sigma = params.sigma_min
    assumed_blur = params.sigma_in
    sigma_diff = sqrt(sigma**2 - assumed_blur**2) / params.delta_min
    image = cv2.GaussianBlur(image, (0, 0), sigma_diff)

    sigmas = []
    deltas = [params.delta_min]
    images = []
    for o in range(params.n_oct):
        if o != 0:
            deltas.append(deltas[-1] * 2)
            sigma = sigmas[o - 1][params.n_spo - 1]  # last valid sigma

        new_sigmas = getSigmas(sigma, params, (deltas[-1]) / params.delta_min)
        sigmas.append(new_sigmas)

        o_images = [image]
        for s, cur_sigma in enumerate(sigmas[o][1:]):
            prev_sigma = sigmas[o][s]
            new_sigma = sqrt(cur_sigma**2 - prev_sigma**2) / deltas[-1]
            image = cv2.GaussianBlur(image, (0, 0), new_sigma)
            o_images.append(image)
        images.append(o_images)
        o_base_image = o_images[params.n_spo - 1]  # image with correct blur
        image = cv2.resize(
            o_base_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
        )

    print("Creating Scale Space... Done!")

    return images, deltas, sigmas


def getSigmas(sigma: float, params: SIFT_Params, mult=1):
    image_count = params.n_spo + 3  # +3 to have enough image for DoG later on

    sigmas = [sigma]
    for i in range(1, image_count):
        c_sigma = mult * params.sigma_min * (2 ** (i / params.n_spo))
        sigmas.append(c_sigma)
    return sigmas


def create_dogs(images: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    print("Calculating Dogs...", end="\r")
    dog_images = []

    for o_images in images:
        dog_o_images = []
        for s in range(len(o_images) - 1):
            dog_o_images.append(cv2.subtract(o_images[s + 1], o_images[s]))
        dog_images.append(dog_o_images)

    print("Calculating Dogs... Done!")
    return dog_images


def find_discrete_extremas(dog_images: list[list[np.ndarray]], border=1, threshold=0):
    print("Searching for Keypoints...", end="\r")
    keypoints = []

    total_steps = len(dog_images) * (len(dog_images[0]) - 2)
    step = 0

    for o, o_images in enumerate(dog_images):
        for s in range(1, len(o_images) - 1):
            printProgress(step, total_steps, "Searching for Keypoints...")
            step += 1

            first_image = o_images[s - 1]
            second_image = o_images[s]
            third_image = o_images[s + 1]
            for x in range(border, first_image.shape[0] - border):
                for y in range(border, first_image.shape[1] - border):
                    first_part = first_image[x - 1 : x + 2, y - 1 : y + 2]
                    second_part = second_image[x - 1 : x + 2, y - 1 : y + 2]
                    third_part = third_image[x - 1 : x + 2, y - 1 : y + 2]
                    if check_local_extremum(
                        first_part, second_part, third_part, threshold
                    ):
                        keypoint = SIFT_KeyPoint(o, s, y, x)
                        keypoints.append(keypoint)

    print("Searching for Keypoints... Done!")
    return keypoints


def check_local_extremum(
    first_part: np.ndarray,
    second_part: np.ndarray,
    third_part: np.ndarray,
    threshold: float,
):
    center = second_part[1, 1]
    if abs(center) < threshold or center == 0:
        return False
    if center > 0:
        return (
            np.all(center >= first_part)
            and np.all(center >= second_part)
            and np.all(center >= third_part)
        )
    else:
        return (
            np.all(center <= first_part)
            and np.all(center <= second_part)
            and np.all(center <= third_part)
        )


# =============================================
# HELPER FUNCTIONS
# =============================================


def gaussFilter(image, size=5):
    kernel = gaussKernel(size)
    return cv2.filter2D(image, -1, kernel)


def gaussKernel(size):
    triangle = pascal(size, kind="lower")  # Pascal triangle
    pRow = triangle[-1]  # last row
    kernel = np.outer(pRow, pRow)  # last row * last row
    return normalizeKernel(kernel)


def normalizeKernel(kernel):
    total = np.sum(kernel)
    return np.multiply(kernel, 1 / total)


def printProgress(i, maxI, msg, endMsg=None):
    i += 1
    maxLen = 20
    curLen = int((i / maxI) * maxLen)
    print(f"{msg} [{'=' * curLen}{' ' * (maxLen-curLen)}]", end="\r")
    if i >= maxI:
        print(" " * (len(msg) + maxLen + 10), end="\r")
        if endMsg:
            print(endMsg)


if __name__ == "__main__":
    main()
