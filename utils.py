import random
from collections import defaultdict
from typing import List, Tuple

from google.cloud import vision
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def read_ocr_from_google_vision_api(img_path: str, load_from_local: bool = False):
    with open(img_path, 'rb') as image_file:
        content = image_file.read()

    if not load_from_local:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
    else:
        with open("response.json", "r") as f:
            response = json.load(f)

    return response


def remove_false_detected_close_horizontal_lines(bounding_boxes: list, vertical: bool = False,
                                                 pixels_threshold: int = 10) -> List:
    indexes_to_remove = []
    vertical = 0 if vertical else 1
    for index, i in enumerate(bounding_boxes):
        if index > 0:

            if abs(i[vertical] - bounding_boxes[index - 1][vertical]) <= pixels_threshold:
                indexes_to_remove.append(index)

    bounding_boxes = [i for index, i in enumerate(bounding_boxes) if index not in indexes_to_remove]

    return bounding_boxes


def detect_lines(img: np.ndarray, detection_type: str, low_threshold: int = 0,
                 kernel_size: int = 15, plot: bool = False, save: bool = False) -> List:
    if img is None:
        raise ValueError("Image not found or path incorrect.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) if detection_type == 'table' \
        else cv2.threshold(gray, low_threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if detection_type == 'table':

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))

        horizontal = cv2.erode(thresh, kernel_h)
        horizontal = cv2.dilate(horizontal, kernel_h)

        vertical = cv2.erode(thresh, kernel_v)
        vertical = cv2.dilate(vertical, kernel_v)

        mask = cv2.add(horizontal, vertical)

    elif detection_type == 'vertical':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        if len(vertical_lines.shape) == 3:
            vertical_lines = cv2.cvtColor(vertical_lines, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(vertical_lines, low_threshold, 255, cv2.THRESH_BINARY)

    elif detection_type == 'horizontal':

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        _, mask = cv2.threshold(horizontal_lines, low_threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda cnt: np.min(cnt[:, 0, 0])) if detection_type != 'horizontal' \
        else sorted(contours, key=lambda cnt: np.min(cnt[:, 0, 1]))

    bounding_boxes = [cv2.boundingRect(cnt) for cnt in sorted_contours]
    if detection_type != 'table':
        is_vertical = True if detection_type == 'vertical' else False
        bounding_boxes = remove_false_detected_close_horizontal_lines(bounding_boxes, vertical=is_vertical)

    img_with_boxes = img.copy()
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if detection_type != 'table':
            cv2.putText(img_with_boxes, str(i), (x + w + 5, y + h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    if plot:
        plt.figure(figsize=(12, 10))
        plt.imshow('img', img_rgb)
        plt.axis("off")
        plt.title(f"Detected {detection_type} Bounding Boxes")

    file_name = f"{detection_type}_image.jpeg"
    path = os.path.join(os.getenv('STAGE_DIR'), file_name)
    cv2.imwrite(path, img_rgb)
    return bounding_boxes


def rect_to_vertices(x, y, w, h):
    return [
        (x, y),  # Top-left
        (x + w, y),  # Top-right
        (x + w, y + h),  # Bottom-right
        (x, y + h)  # Bottom-left
    ]


def polygon_to_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def box_contains(box_outer, box_inner):
    x2_min, y2_min, x2_max, y2_max = polygon_to_bbox(box_inner)

    for key, table in box_outer.items():
        x1_min, y1_min, x1_max, y1_max = polygon_to_bbox(table[0])

        if (x1_min <= x2_min and x1_max >= x2_max and
                y1_min <= y2_min and y1_max >= y2_max):
            box_outer[key] = (table[0], table[1] + 1)

    return box_outer


def draw_box(img: np.ndarray, vertices, color: Tuple[int] = (0, 255, 0), thickness: int = 1,
             font_scale: int = 2) -> None:
    points = [(v.x, v.y) for v in vertices]
    pts = np.array(points, dtype=np.int32)

    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    if len(points) >= 1:
        x, y = points[2]
        cv2.putText(img, f'({x}, {y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)


def get_max_table(table_bounding_boxes_dict: dict, not_good_max_keys:List[int]) -> int:
    max_val, max_key = 0, None
    for key, value in table_bounding_boxes_dict.items():
        if value[1] > max_val and key not in not_good_max_keys:
            max_val = value[1]
            max_key = key
    return max_key


def plot_table_bounding_boxes(img: np.ndarray, table):
    pts = np.array(table[0], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=3)


def get_table_with_most_words(img: np.ndarray, response, table_bounding_boxes_dict: dict, plot: bool = False,
                              not_good_max_keys: List[int] = []) -> int:
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_ver = [(v.x, v.y) for v in word.bounding_box.vertices]
                    table_bounding_boxes_dict = box_contains(table_bounding_boxes_dict, word_ver)
                    word_text = "".join([s.text for s in word.symbols])
                    if plot:
                        draw_box(img, word.bounding_box.vertices, label='word', color=(0, 100, 200), font_scale=0.4)

    # get table with max words inside it
    max_key = get_max_table(table_bounding_boxes_dict, not_good_max_keys)

    if plot:
        plot_table_bounding_boxes(img, table_bounding_boxes_dict[max_key])

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return max_key


def compare(outer_box, inner_box):
    x2_min, y2_min, x2_max, y2_max = polygon_to_bbox(inner_box)
    x1_min, y1_min, x1_max, y1_max = polygon_to_bbox(outer_box)

    return (x1_min <= x2_min and x1_max >= x2_max and
            y1_min <= y2_min and y1_max >= y2_max)


def get_all_relevant_words_from_table(response, table_bounding_boxes_dict: dict, max_key: int):
    rellevant_words, rellevant_words_symbols = list(), list()
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_ver = [(v.x, v.y) for v in word.bounding_box.vertices]
                    if compare(table_bounding_boxes_dict[max_key][0], word_ver):
                        rellevant_words.append(word_ver)
                        rellevant_words_symbols.append(''.join([symbol.text for symbol in word.symbols]))

    return rellevant_words, rellevant_words_symbols


def get_cropped_image_offset(polygon) -> Tuple[int, int]:
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    x_min, y_min = min(xs), min(ys)
    return x_min, y_min


def crop_image_by_polygon(img: np.ndarray, polygon, plot: str = False, save: bool = False):
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cropped = img[y_min:y_max, x_min:x_max]
    if plot:
        cv2.imshow('cropped', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        save_path = f'{os.getenv("STAGE_DIR")}/table_detection.jpeg'
        cv2.imwrite(save_path, cropped)

    return cropped


def plot_sorted_contours(img: np.ndarray, sorted_contours) -> None:
    for i, cnt in enumerate(sorted_contours):
        x, y, w, h = cnt[0], cnt[1], cnt[2], cnt[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, str(i), (x + w + 5, y + h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def get_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def word_location_between_lines(img: np.ndarray, relevant_words, bounding_boxes, vertical=True, x0=0, y0=0, plot=False):
    x_locations = {}

    sorted_contours = [tuple(i + np.array([x0, y0, 0, 0])) for i in bounding_boxes]
    sorted_contours = [tuple(int(x) for x in item) for item in sorted_contours]

    min_positions = [int(np.min(cnt[0])) for cnt in sorted_contours] if vertical \
        else [int(np.min(cnt[1])) for cnt in sorted_contours]

    for word_index, word in enumerate(relevant_words):

        target_x = (word[0][0] + word[2][0]) // 2 if vertical \
            else (word[0][1] + word[2][1]) // 2

        for i in range(len(min_positions) - 1):
            if min_positions[i] <= target_x <= min_positions[i + 1]:
                x_locations[word_index] = (str(f"{i}-{i + 1}"), word)
                # if plot:
                if vertical:
                    cv2.putText(img, str(f"{i}-{i + 1}"), (target_x + 5, word[2][1] + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                else:
                    cv2.putText(img, str(f"{i}-{i + 1}"), (word[2][0] + 2, target_x + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                break

        # if plot:
        pts = np.array(word, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

    plot_sorted_contours(img, sorted_contours)
    if plot:
        cv2.imshow('texts', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    file_name = f"vertical_words_image.jpeg" if vertical else f"horizontal_words_image.jpeg"
    path = os.path.join(os.getenv('STAGE_DIR'), file_name)
    cv2.imwrite(path, img)

    return x_locations


def create_grouped_words_by_row_or_col(img: np.ndarray, x_locations, plot=False):
    merged = defaultdict(list)

    for key, value in x_locations.items():
        merged[value[0]].append(value[1])

    for i, val in merged.items():
        color = get_random_color()
        for box in val:
            if plot:
                pts = np.array(box, dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)

    if plot:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return merged


def merge_boxes(boxes):
    all_points = [point for box in boxes for point in box]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    return [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]


def create_df_for_words(horizontal_x_locations, vertical_x_locations, rellevant_words_symbols):
    data = []
    for index, (hor, vert, word) in enumerate(
            zip(horizontal_x_locations.values(), vertical_x_locations.values(), rellevant_words_symbols)):
        data.append(tuple([index, hor[0], vert[0], vert[1], word]))

    df = pd.DataFrame(data, columns=['index', 'horizontal_label', 'vertical_label', 'box', 'word'])

    grouped = df.groupby(['horizontal_label', 'vertical_label'])

    merged = grouped.agg({
        'word': lambda x: ' '.join(map(str, x)),
        'box': lambda boxes: merge_boxes(boxes)
    }).reset_index()
    merged['horizontal_order'] = merged['horizontal_label'].apply(lambda x: int(x.split('-')[0]))
    merged['vertical_order'] = merged['vertical_label'].apply(lambda x: int(x.split('-')[0]))
    merged.sort_values(by=['horizontal_order', 'vertical_order'])
    return merged


def plot_df_words(img: np.ndarray, df, plot=False):
    for index, row in df.iterrows():
        pts = np.array(row['box'], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=1)
    if plot:
        cv2.imshow('words', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    file_name = f"full_words_image.jpeg"
    path = os.path.join(os.getenv('STAGE_DIR'), file_name)
    cv2.imwrite(path, img)


def validate_table_df(table_df: pd.DataFrame) -> pd.DataFrame:
    for col in table_df.columns:
        try:
            table_df[col] = pd.to_numeric(table_df[col])
        except:
            print(f'col {col} cannot be converted to a number')
    table_df.fillna(0, inplace=True)
    return table_df


def convert_df_to_table(df, remove_last_rows=0) -> pd.DataFrame:
    rows = max(df['horizontal_order']) + 1
    cols = max(df['vertical_order']) + 1

    matrix = [['' for _ in range(cols)] for _ in range(rows)]

    df.sort_values(by=['vertical_order', 'horizontal_order'], inplace=True)

    for _, row in df.iterrows():
        if row['vertical_label'] not in ['10-11', '9-10']:
            row_ind = int(row['horizontal_order'])
            col_ind = int(row['vertical_order'])
            matrix[row_ind][col_ind] = row['word'].replace('|', '').replace('*', '')

    table_df = pd.DataFrame(matrix[1:], columns=matrix[0])
    table_df = table_df.iloc[:-remove_last_rows]
    return validate_table_df(table_df)
