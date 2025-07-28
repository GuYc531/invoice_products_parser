from typing import Tuple

import cv2
import pandas as pd

from backend import utils as u
from dotenv import load_dotenv
import os

load_dotenv()


class Parser:
    def __init__(self):
        self.horizontal_x_locations = None
        self.vertical_x_locations = None
        self.relevant_words_symbols = None
        self.relevant_words = None
        self.uploaded_image_path = None
        self.img = None
        self.response = None
        self.max_key = None
        self.cropped = None
        self.vertical_bounding_boxes = None
        self.horizontal_bounding_boxes = None
        self.table_image_path = None
        self.table_bounding_boxes_dict = None
        self.max_keys_not_valid = []

    def inserted_uploaded_image_path(self, uploaded_image_path: str) -> None:
        self.uploaded_image_path = uploaded_image_path
        self.img = cv2.imread(self.uploaded_image_path)

    def get_response_from_api(self) -> None:
        self.response = u.read_ocr_from_google_vision_api(self.uploaded_image_path)

    def get_table_bounding_boxes(self) -> str:
        table_bounding_boxes = u.detect_lines(self.img, detection_type='table', low_threshold=0, kernel_size=30)
        self.table_bounding_boxes_dict = {index: (u.rect_to_vertices(i[0], i[1], i[2], i[3]), 0) \
                                          for index, i in enumerate(table_bounding_boxes)}
        self.max_key = u.get_table_with_most_words(self.img, self.response, self.table_bounding_boxes_dict,
                                                   not_good_max_keys=self.max_keys_not_valid)
        self.max_keys_not_valid.append(self.max_key)
        self.cropped = u.crop_image_by_polygon(self.img, self.table_bounding_boxes_dict[self.max_key][0], save=True)
        file_name = 'table_detection.jpeg'
        self.table_image_path = os.path.join(os.getenv('STAGE_DIR'), file_name)
        return file_name

    def get_lines(self, vertical: bool = True) -> str:
        image = cv2.imread(self.table_image_path)
        if vertical:
            self.vertical_bounding_boxes = u.detect_lines(image, detection_type='vertical', low_threshold=0,
                                                          kernel_size=12)
            return 'vertical_image.jpeg'
        else:
            self.horizontal_bounding_boxes = u.detect_lines(image, detection_type='horizontal', low_threshold=0,
                                                            kernel_size=70)
            return 'horizontal_image.jpeg'

    def get_words_locations(self, vertical=True) -> str:
        self.relevant_words, self.relevant_words_symbols = u.get_all_relevant_words_from_table(
            self.response, self.table_bounding_boxes_dict, self.max_key)
        x0, y0 = u.get_cropped_image_offset(self.table_bounding_boxes_dict[self.max_key][0])
        img = cv2.imread(self.uploaded_image_path)

        if vertical:
            self.vertical_x_locations = u.word_location_between_lines(img=img, relevant_words=self.relevant_words,
                                                                      bounding_boxes=self.vertical_bounding_boxes,
                                                                      x0=x0, y0=y0)
            return 'vertical_words_image.jpeg'
        else:
            self.horizontal_x_locations = u.word_location_between_lines(img=img, vertical=False,
                                                                        relevant_words=self.relevant_words,
                                                                        bounding_boxes=self.horizontal_bounding_boxes,
                                                                        x0=x0, y0=y0)
            return 'horizontal_words_image.jpeg'


    def get_words_df(self) -> Tuple[str, pd.DataFrame]:
        df = u.create_df_for_words(self.horizontal_x_locations, self.vertical_x_locations, self.relevant_words_symbols)
        img = cv2.imread(self.uploaded_image_path)

        u.plot_df_words(img, df)

        final_df = u.convert_df_to_table(df, remove_last_rows=1)
        return 'full_words_image.jpeg', final_df
