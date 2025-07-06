import cv2
import utils as u
from dotenv import load_dotenv

load_dotenv()

load_from_local = False
PATH_TO_IMAGE = 'assets/invo.jpeg'

img = cv2.imread(PATH_TO_IMAGE)

response = u.read_ocr_from_google_vision_api(PATH_TO_IMAGE)

PLOT = False
table_bounding_boxes = u.detect_lines(img, detection_type='table', low_threshold=0, kernel_size=30, plot=PLOT)
table_bounding_boxes_dict = {index: (u.rect_to_vertices(i[0], i[1], i[2], i[3]), 0) \
                             for index, i in enumerate(table_bounding_boxes)}
max_key = u.get_table_with_most_words(img, response, table_bounding_boxes_dict, plot=PLOT)
cropped = u.crop_image_by_polygon(img, table_bounding_boxes_dict[max_key][0], plot=PLOT)

vertical_bounding_boxes = u.detect_lines(cropped, detection_type='vertical', low_threshold=0, kernel_size=12, plot=PLOT)
horizontal_bounding_boxes = u.detect_lines(cropped, detection_type='horizontal', low_threshold=0, kernel_size=70,
                                           plot=PLOT)

img = cv2.imread(PATH_TO_IMAGE)

relevant_words, relevant_words_symbols = u.get_all_relevant_words_from_table(response, table_bounding_boxes_dict, max_key)
x0, y0 = u.get_cropped_image_offset(table_bounding_boxes_dict[max_key][0])

vertical_x_locations = u.word_location_between_lines(img=img, relevant_words=relevant_words,
                                                     bounding_boxes=vertical_bounding_boxes,
                                                     x0=x0, y0=y0, plot=PLOT)

img = cv2.imread(PATH_TO_IMAGE)

horizontal_x_locations = u.word_location_between_lines(img=img, vertical=False, relevant_words=relevant_words,
                                                       bounding_boxes=horizontal_bounding_boxes,
                                                       x0=x0, y0=y0, plot=PLOT)

df = u.create_df_for_words(horizontal_x_locations, vertical_x_locations, relevant_words_symbols)
img = cv2.imread(PATH_TO_IMAGE)

u.plot_df_words(img, df, plot=PLOT)

vertical_x_locations_grouped = u.create_grouped_words_by_row_or_col(img, vertical_x_locations, plot=PLOT)
horizontal_x_locations_grouped = u.create_grouped_words_by_row_or_col(img, horizontal_x_locations, plot=PLOT)

final_df = u.convert_df_to_table(df, remove_last_rows=1)
