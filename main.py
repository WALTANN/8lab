import time
import os
import cv2
import numpy as np

def write_coordinates_to_file(x_coordinate=int, y_coordinate=int):
    with open('point_coordinates.txt', 'a') as file_with_coordinates:
        file_with_coordinates.write(f'Coordinates: x = {x_coordinate}, y = {y_coordinate}\n')


def gaussian_blur():
    readed_image = cv2.imread('variant-2.png', cv2.IMREAD_COLOR)
    readed_image_do_gaussianblur = cv2.GaussianBlur(readed_image, (15, 15), 0)
    cv2.imwrite('image_with_gaussianblur_variant_2.png', readed_image_do_gaussianblur)
    print('"image_with_gaussianblur_variant_2" создана!')


def clear_folder():
    if os.path.isfile('image_with_gaussianblur.png'):
        os.remove('image_with_gaussianblur.png')
    else:
        print('no')


def video_processing():
    print('Нажми q чтобы завершить!')
    all_x_coordinates = [1]
    all_y_cooordinates = [1]
    THRESHOLD = 0.7
    capture = cv2.VideoCapture(0)
    point_to_search = cv2.imread('ref-point.jpg', 0)
    h, w = point_to_search.shape
    while True:
        _, moment_frame = capture.read()

        moment_frame_TurnIntoGrayColor = cv2.cvtColor(moment_frame, cv2.COLOR_RGB2GRAY)

        res = cv2.matchTemplate(moment_frame_TurnIntoGrayColor, point_to_search, cv2.TM_CCOEFF_NORMED)

        object_location = np.where(res >= THRESHOLD)

        for coordinates in zip(*object_location[::-1]):
            cv2.circle(moment_frame, (coordinates[0] + h // 2, coordinates[1] + w // 2), h // 2, (0, 0, 0), 1)

            print(f'центр: ({coordinates[0] + h // 2}, {coordinates[1] + w // 2})')

            write_coordinates_to_file(int(coordinates[0]), int(coordinates[1]))

            all_x_coordinates.append(coordinates[0] + h // 2)
            all_y_cooordinates.append(coordinates[1] + w // 2)

        cv2.imshow('point_search', moment_frame)

        quit_button = cv2.waitKey(1)
        if quit_button == ord('q'):
            print(
                f'Среднее значение центра фигуры:\nx = {int(sum(all_x_coordinates[1:]) / len(all_x_coordinates[1:]))}\ny = {int(sum(all_y_cooordinates[1:]) / len(all_y_cooordinates[1:]))}')

            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    gaussian_blur()
    video_processing()
    # clear_folder()