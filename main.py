import pygame as pg
import numpy as np
import pygame.surfarray
from numba import njit
import cv2 as cv
import mediapipe as mp
import pyvirtualcam
from sys import exit
from enum import Enum


@njit(fastmath=True)
def accelerate_conversion(image, gray_image, width, height, color_coeff, ascii_coeff, step):
    array_of_values = []
    for x in range(0, width, step):
        for y in range(0, height, step):
            char_index = gray_image[x, y] // ascii_coeff
            if char_index:
                r, g, b = image[x, y] // color_coeff
                array_of_values.append((char_index, (r, g, b), (x, y)))
    return array_of_values


class CameraMode(Enum):
    NORMAL = 1
    FULL_ASCII = 2
    PERSON_ASCII = 3
    BACKGROUND_ASCII = 4


class VideoConverter:
    def __init__(self, font_size=12, color_lvl=8):
        pg.init()
        self.capture = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.COLOR_LVL = color_lvl
        self.image, self.gray_image = self.get_image()
        self.cv_image = None
        self.RES = self.WIDTH, self.HEIGHT = self.image.shape[0], self.image.shape[1]
        self.bg_image = cv.transpose(np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8))
        self.surface = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        self.cam = pyvirtualcam.Camera(width=self.WIDTH, height=self.HEIGHT, fps=25, fmt=pyvirtualcam.PixelFormat.BGR)

        self.remove_background = False

        self.camera_mode = CameraMode.NORMAL

        self.ASCII_CHARS = ' _.,-=+:;cba!?0123456789$W#@Ñ'
        self.ASCII_COEFF = 255 // (len(self.ASCII_CHARS) - 1)

        self.font = pg.font.SysFont('Courier', font_size, bold=True)
        self.CHAR_STEP = int(font_size * 0.6)
        self.PALETTE, self.COLOR_COEFF = self.create_palette()

        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def get_frame(self):
        frame = pg.surfarray.array3d(self.surface)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        return cv.transpose(frame)

    def draw_converted_image(self):
        image, gray_image = self.get_image()

        if self.remove_background:
            # get the results
            results = self.selfie_segmentation.process(image)

            condition = np.stack((results.segmentation_mask) * 3, axis=-1) < 0.5

            falses = np.transpose(condition)

            image[falses] = 0
            gray_image[falses] = 0

        if self.camera_mode == CameraMode.NORMAL:
            self.surface.blit(pg.surfarray.make_surface(image), (0, 0))
        elif self.camera_mode == CameraMode.FULL_ASCII:
            array_of_values = accelerate_conversion(image, gray_image, self.WIDTH, self.HEIGHT, self.COLOR_COEFF,
                                                    self.ASCII_COEFF, self.CHAR_STEP)

            for char_index, color, pos in array_of_values:
                char = self.ASCII_CHARS[char_index]
                self.surface.blit(self.PALETTE[char][color], pos)
        elif self.camera_mode == CameraMode.PERSON_ASCII:
            array_of_values = accelerate_conversion(image, gray_image, self.WIDTH, self.HEIGHT, self.COLOR_COEFF,
                                                    self.ASCII_COEFF, self.CHAR_STEP)

            for char_index, color, pos in array_of_values:
                char = self.ASCII_CHARS[char_index]
                self.surface.blit(self.PALETTE[char][color], pos)
            results = self.selfie_segmentation.process(image)

            condition = np.stack((results.segmentation_mask) * 3, axis=-1) > 0.5

            falses = np.transpose(condition)
            image[falses] = 0

            sur = pg.surfarray.make_surface(image)
            sur.set_colorkey((0, 0, 0))

            self.surface.blit(sur, (0, 0))
        elif self.camera_mode == CameraMode.BACKGROUND_ASCII:
            array_of_values = accelerate_conversion(image, gray_image, self.WIDTH, self.HEIGHT, self.COLOR_COEFF,
                                                    self.ASCII_COEFF, self.CHAR_STEP)

            for char_index, color, pos in array_of_values:
                char = self.ASCII_CHARS[char_index]
                self.surface.blit(self.PALETTE[char][color], pos)
            results = self.selfie_segmentation.process(image)

            condition = np.stack((results.segmentation_mask) * 3, axis=-1) < 0.5

            falses = np.transpose(condition)
            image[falses] = 0

            sur = pg.surfarray.make_surface(image)
            sur.set_colorkey((0, 0, 0))

            self.surface.blit(sur, (0, 0))

    def create_palette(self):
        colors, color_coeff = np.linspace(0, 255, num=self.COLOR_LVL, dtype=int, retstep=True)
        color_palette = [np.array([r, g, b]) for r in colors for g in colors for b in colors]
        palette = dict.fromkeys(self.ASCII_CHARS, None)
        for char in palette:
            char_palette = {}
            for color in color_palette:
                color_key = tuple(color // color_coeff)
                char_palette[color_key] = self.font.render(char, False, tuple(color))
            palette[char] = char_palette
        return palette, color_coeff

    def get_image(self):
        ret, self.cv_image = self.capture.read()
        if not ret:
            exit()
        transposed_image = cv.transpose(self.cv_image)
        image = cv.cvtColor(transposed_image, cv.COLOR_BGR2RGB)
        gray_image = cv.cvtColor(transposed_image, cv.COLOR_BGR2GRAY)
        return image, gray_image

    def draw(self):
        self.surface.fill('black')
        self.draw_converted_image()

    def stream_camera(self):
        frame = self.get_frame()
        self.cam.send(frame)
        self.cam.sleep_until_next_frame()

    def run(self):
        while True:
            for i in pg.event.get():
                if i.type == pg.QUIT:
                    exit()
                elif i.type == pg.KEYDOWN:
                    if i.key == pg.K_b:
                        self.remove_background = not self.remove_background
                    elif i.key == pg.K_n:
                        self.camera_mode = CameraMode.NORMAL
                    elif i.key == pg.K_f:
                        self.camera_mode = CameraMode.FULL_ASCII
                    elif i.key == pg.K_p:
                        self.camera_mode = CameraMode.PERSON_ASCII
                    elif i.key == pg.K_a:
                        self.camera_mode = CameraMode.BACKGROUND_ASCII
            self.stream_camera()
            self.draw()
            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.flip()
            self.clock.tick()


if __name__ == '__main__':
    app = VideoConverter()
    app.run()
