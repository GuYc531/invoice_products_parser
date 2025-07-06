import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store user feedback
user_feedback = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    image_filename = None

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                image_filename = image.filename
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                image.save(save_path)
        if 'feedback' in request.form:
            image_filename = request.form.get("image_name")
            feedback = request.form.get("feedback")
            user_feedback[image_filename] = feedback
            print(f"📥 Feedback for {image_filename}: {feedback}")
            return redirect(url_for('index'))

    # Show the last uploaded image or fallback
    image_to_display = image_filename or get_latest_uploaded_image() or "fallback.jpg"
    return render_template("index.html", image_name=image_to_display)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_latest_uploaded_image():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return files[-1] if files else None

if __name__ == '__main__':
    app.run(debug=True)
