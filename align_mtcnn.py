import cv2 as cv
import numpy as np
import pandas as pd
from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector
import os
import cv2

def process(img, output_size):
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
    cv.imwrite('azim_images/{}_mtcnn_aligned_{}x{}.jpg'.format(az, output_size[0], output_size[1]), dst_img)
    # img = cv.resize(raw, (224, 224))
    # cv.imwrite('images/{}_img.jpg'.format(i), img)


if __name__ == "__main__":
    detector = MtcnnDetector()

    image_rows, image_columns, image_depth = 64, 64, 18
    flow_rows, flow_columns, flow_depth = 144, 120, 16

    SAMM_dataset_path = '/home/bsft19/mohamazim2/Windows/Downloads/SAMM/'
    # dataSheet = pandas.read_excel(io= 'C:/Users/mohamazim2/Downloads/excel/xcel.xlsx', usecols= 'B, D:F, G, J', index_col = 0)
    dataSheet = pd.read_excel(io="/home/bsft19/mohamazim2/Windows/Downloads/excel/xcel.xlsx",
                                  usecols='B, D:F, G, J', index_col=0)
    SAMM_list = []
    SAMM_labels = []
    SAMM_flow_list = []
    directorylisting = os.listdir(SAMM_dataset_path)
    SAMM_subject_boundary = []
    SAMM_subject_boundary.append(0)
    print(dataSheet)
    # az = 0
    print(directorylisting)

    for subject in directorylisting[1:2]:

        boundary = SAMM_subject_boundary[-1]

        print("SAMM current subject: ", subject)

        for sample in os.listdir(SAMM_dataset_path + subject + '/'):

            apex = dataSheet.loc[sample, 'Apex Frame']

            emotion = dataSheet.loc[sample, 'Estimated Emotion']

            onset = dataSheet.loc[sample, 'Onset Frame']

            offset = dataSheet.loc[sample, 'Offset Frame']

            frames = []

            frame_count = 18

            height = 640

            width = 960

            video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')

            if (dataSheet.loc[sample, 'Duration'] >= image_depth and emotion != 'Other'):

                label = 0

                if (emotion == 'Happiness'):
                    label = 1
                elif (emotion == 'Surprise'):
                    label = 2

                count = 0

                start = 0

                end = 0

                if (apex - onset > (image_depth / 2) and offset - apex >= (image_depth / 2)):

                    start = apex - (image_depth / 2)

                    end = apex + (image_depth / 2)

                elif (apex - onset <= (image_depth / 2)):

                    start = onset

                    end = onset + image_depth - 1

                elif (offset - apex < (image_depth / 2)):

                    start = offset - image_depth

                    end = offset
                az = 0
                for image in os.listdir(SAMM_dataset_path + subject + '/' + sample + '/'):

                    imagecode = int(image[-4 - len(str(apex)):-4])

                    if (imagecode >= start and imagecode < end):
                        image_path = SAMM_dataset_path + subject + '/' + sample + '/' + image
                        # img = cv2.imread(image_path)
                        # filename = 'images/{}_raw.jpg'.format(az)
                        # print('Loading image {}'.format(filename))
                        print(image_path)
                        raw = cv.imread(image_path)
                        # print("RAW")
                        print(raw)
                        cv2.imwrite('azim_images/{}_mtcnn_aligned_{}x{}.jpg'.format(az,'original','one'), raw)
                        # print('should sace')
                        process(raw, output_size=(224, 224))
                        # process(raw, output_size=(112, 112))
                        # frames.append(img)
                        # temp = img[:640, :, :]
                        # video_tensor[count] = temp
                        #
                        # count = count + 1
                    az += 1

    # for i in range(10):
    #     filename = 'images/{}_raw.jpg'.format(i)
    #     print('Loading image {}'.format(filename))
    #     raw = cv.imread(filename)
    #     process(raw, output_size=(224, 224))
    #     process(raw, output_size=(112, 112))
