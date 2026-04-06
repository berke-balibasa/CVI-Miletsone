import random
import cv2
import numpy as np


def read_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image could not be loaded: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def random_flip(image, steering):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering
    return image, steering


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    factor = 0.5 + np.random.rand()
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image


def random_zoom(image):
    h, w = image.shape[:2]
    scale = random.uniform(1.0, 1.2)

    new_h = int(h / scale)
    new_w = int(w / scale)

    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)

    cropped = image[top:top + new_h, left:left + new_w]
    zoomed = cv2.resize(cropped, (w, h))

    return zoomed


def random_pan(image, steering):
    h, w = image.shape[:2]

    x_shift = random.randint(-40, 40)
    y_shift = random.randint(-10, 10)

    matrix = np.float32([
        [1, 0, x_shift],
        [0, 1, y_shift]
    ])

    panned = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    steering += x_shift * 0.002

    return panned, steering


def random_rotate(image, steering):
    h, w = image.shape[:2]

    angle = random.uniform(-5, 5)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    steering += angle * 0.02

    return rotated, steering


def augment_image(image, steering):
    image, steering = random_flip(image, steering)

    if random.random() < 0.5:
        image = random_brightness(image)

    if random.random() < 0.5:
        image = random_zoom(image)

    if random.random() < 0.5:
        image, steering = random_pan(image, steering)

    if random.random() < 0.5:
        image, steering = random_rotate(image, steering)

    steering = np.clip(steering, -1, 1)

    return image, steering


def pre_processing(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255.0

    return image


def batch_generator(image_paths, steering_values, batch_size, is_training):
    while True:
        batch_images = []
        batch_steering = []

        for _ in range(batch_size):
            index = random.randint(0, len(image_paths) - 1)

            image = read_image(image_paths[index])
            steering = steering_values[index]

            if is_training:
                image, steering = augment_image(image, steering)

            image = pre_processing(image)

            batch_images.append(image)
            batch_steering.append(steering)

        yield np.asarray(batch_images), np.asarray(batch_steering)