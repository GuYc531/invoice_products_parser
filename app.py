import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file
import os
import io
from backend.Parser import Parser

app = Flask(__name__)
UPLOAD_DIR = os.getenv('UPLOAD_DIR')
STAGE_DIR = os.getenv('STAGE_DIR')
DATA_DIR = os.getenv('DATA_DIR')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

stage_sentences = {
    1: 'Please upload an invoice image in jpeg/jpg format',
    2: 'Here is the uploaded image, continue with Approve',
    3: '''Is this your main table of products?
            is so approve, if not reject and will try another table''',
    4: 'Is this the vertical lines of the table?',
    5: 'Is this the horizontal lines of the table?',
    6: 'Is this the words locations based on the horizontal lines? ',
    7: 'Is this the words locations based on the vertical lines? ',
    8: '''Please check the table from the right to see your products in a tabel format.
     You can download to csv if needed'''
}


current_stage = 1
stage_images = {}
parser = Parser()
df = pd.DataFrame()


@app.route('/', methods=['GET', 'POST'])
def index():
    global current_stage, stage_images, parser, df

    if request.method == 'POST':
        if 'reset_action' in request.form and current_stage == 8:
            print(f"resets system to another uploaded image")
            current_stage = 1
            stage_images = {}
            parser = Parser()

        if current_stage == 1 and 'image' in request.files:
            image = request.files['image']
            if image.filename:
                image_path = os.path.join(UPLOAD_DIR, 'uploaded.jpg')
                image.save(image_path)
                stage_images[1] = 'uploaded.jpg'
                current_stage = 2
                parser.inserted_uploaded_image_path(image_path)
                print("saved uploaded image")

        elif 'action' in request.form:
            action = request.form['action']
            if action == 'approve':
                if current_stage == 2:
                    parser.get_response_from_api()
                    stage_images[2] = parser.get_table_bounding_boxes()
                    current_stage = 3
                elif current_stage == 3:
                    stage_images[3] = parser.get_lines(vertical=True)
                    current_stage = 4
                elif current_stage == 4:
                    stage_images[4] = parser.get_lines(vertical=False)
                    current_stage = 5
                elif current_stage == 5:
                    stage_images[5] = parser.get_words_locations(vertical=False)
                    current_stage = 6
                elif current_stage == 6:
                    stage_images[6] = parser.get_words_locations(vertical=True)
                    current_stage = 7
                elif current_stage == 7:
                    stage_images[7], final_df = parser.get_words_df()
                    final_df.to_csv(f'{os.path.join(DATA_DIR, "final_df.csv")}')
                    current_stage = 8
                    df = final_df.copy()

            elif action == 'reject':
                if current_stage == 3:
                    stage_images[2] = parser.get_table_bounding_boxes()

                elif current_stage > 2:
                    current_stage -= 1

        return redirect(url_for('index'))

    if current_stage == 2:
        image = stage_images[1]
        folder = UPLOAD_DIR
    elif 2 < current_stage < 9:
        image = stage_images[current_stage - 1]
        folder = STAGE_DIR
    else:
        image = None
        folder = ""
        print(f"no image uploaded yet")

    return render_template('index.html',
                           stage=current_stage,
                           image=image,
                           folder=folder,
                           table_data=df,
                           sentence=stage_sentences.get(current_stage, "Unknown stage.")
                           )


@app.route('/display/<folder>/<filename>')
def display_image(folder, filename):
    print(f"showing image {os.path.join(folder, filename)}")
    return send_from_directory(folder, filename)


def get_final_dataframe():
    if os.path.exists(f'{os.path.join(DATA_DIR, "final_df.csv")}'):
        return pd.read_csv(f'{os.path.join(DATA_DIR, "final_df.csv")}')
    else:
        print("there is no data frame saved in dir")
        return pd.DataFrame()


@app.route('/download_csv')
def download_csv():
    df = get_final_dataframe()
    csv_buffer = io.StringIO()
    csv_buffer.write("\ufeff")
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)

    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv ; charset=utf-8',
        as_attachment=True,
        download_name='result.csv'
    )


if __name__ == '__main__':
    app.run(debug=False)

