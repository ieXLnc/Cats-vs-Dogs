import os
from flask import Flask, redirect, url_for, render_template, request, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from predict_img import make_predictions

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1000 * 1000


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_img():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No files uploaded")
            return redirect(request.url)
        file = request.files["file"]
        # if no file uploaded, the browser submits an empty file without name
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            # make predictions
            extension_file = "." in filename and filename.rsplit(".", 1)[1].lower()
            label, prob = make_predictions(upload_image_path, extension_file)
            return render_template(
                "classification.html",
                image_file_name=file.filename,
                label=label,
                prob=prob,
            )

    return render_template("home_page.html")


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run()
