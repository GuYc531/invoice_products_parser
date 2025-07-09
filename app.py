from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from Parser import Parser
app = Flask(__name__)
UPLOAD_DIR = 'uploads'
STAGE_DIR = 'saved_images'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STAGE_DIR, exist_ok=True)

# TODO: 1. how to show table in UI?
# TODO: 2. adjust table detection to be requersive if user choses no
# TODO: 3. clean code
# TODO: 4. need to let the user chose how many last rows to remove from table df

current_stage = 1
stage_images = {}
parser = Parser()

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_stage, stage_images

    if request.method == 'POST':
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
                    current_stage = 8

            elif action == 'reject':
                if current_stage > 2:
                    current_stage -= 1

        return redirect(url_for('index'))

    # Choose image and folder for current stage
    if current_stage == 2:
        image = stage_images[1]
        folder = UPLOAD_DIR

    elif current_stage == 3:
        image = stage_images[2]
        folder = STAGE_DIR

    elif current_stage == 4:
        image = stage_images[3]
        folder = STAGE_DIR
    elif current_stage == 5:
        image = stage_images[4]
        folder = STAGE_DIR
    elif current_stage == 6:
        image = stage_images[5]
        folder = STAGE_DIR
    elif current_stage == 7:
        image = stage_images[6]
        folder = STAGE_DIR
    elif current_stage == 8:
        image = stage_images[7]
        folder = STAGE_DIR
    else:
        image = None
        folder = ""
        print(f"no image uploaded yet")


    return render_template('index.html',
                           stage=current_stage,
                           image=image,
                           folder=folder)

@app.route('/display/<folder>/<filename>')
def display_image(folder, filename):
    print(f"showing image {os.path.join(folder, filename)}")
    return send_from_directory(folder, filename)


if __name__ == '__main__':
    app.run(debug=False)
