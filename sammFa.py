import cv2 as cv
import numpy as np
import pandas as pd
from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector
import os
import cv2
from pathlib import Path


def process(img, output_size, path_to_save):
    _, facial5points = detector.detect_faces(img)
    facial5points = np.reshape(facial5points[0], (2, 5))

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=output_size)
    # cv.imwrite('azim_images/{}_mtcnn_aligned_{}x{}.jpg'.format(az, output_size[0], output_size[1]), dst_img)
    cv.imwrite(path_to_save, dst_img)
    # img = cv.resize(raw, (224, 224))
    # cv.imwrite('images/{}_img.jpg'.format(i), img)



def loadData():
    dataSheet = pd.read_excel(io='C:/Users/mohamazim2/Downloads/excel/xcel.xlsx', usecols='B, D:F, G, J',
                                  index_col=0)
    print(dataSheet)



if __name__ == '__main__':
    loadData()
#     detector = MtcnnDetector()
#     file_path = "E:/unzipped_cs229/SAMM/SAMM/SAMM"
#     directory_listing = os.listdir(file_path)
#     new_file_path = "E:/unzipped_cs229/SAMM/SAMM/Samm_with_fa"
#     # print(directory_listing)
#     for img_folder in directory_listing:
#         img_folder_listing = file_path + "/" + img_folder + "/"
#
#         new_img_folder_listing = new_file_path + "/" + img_folder + "/"
#
#         img_directory_listing = os.listdir(img_folder_listing)
#         print(img_directory_listing)
#         for img_files_folder in img_directory_listing:
#             every_image = img_folder_listing + img_files_folder
#
#             new_every_image = new_img_folder_listing + img_files_folder
#
#             images = os.listdir(every_image)
#             # print(images)
#             # try:
#             #     Path(new_every_image).mkdir(parents=True, exist_ok=True)
#             # except Exception as e:
#             #     print("Cannot create")
#             for img in images:
#                 every_image_in_folder = every_image + "/" + img
#                 new_every_image_in_folder = new_every_image + "/" + img
#                 print(every_image_in_folder)
#                 raw = cv.imread(every_image_in_folder)
#                 process(raw, output_size=(224, 224), path_to_save = new_every_image_in_folder)
# # do_Fa()

