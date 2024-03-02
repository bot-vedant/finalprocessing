import os
from flask import Flask, render_template, redirect, request,send_file,Response,url_for,jsonify
import cv2
import numpy as np
from io import BytesIO
import base64
import zipfile

app = Flask(__name__)

    # Given physical dimensions (in mm)
    # desired_width_mm = 21.138
    # desired_height_mm = 23.243
    # estimated_dpi = 300

    # Calculate the scaling factors
    # pixels_per_mm_width = estimated_dpi / 25.4
    # pixels_per_mm_height = estimated_dpi / 25.4
input_image_names = []
processed_images = []
errors = []       
@app.route('/', methods=['GET', 'POST'])
def upload_files():
        result = [] 
        resultimage = []
        
        uploads= []

        if request.method == 'POST':
            uploaded_files = request.files.getlist('file')
            selected_size = request.form['size']
            standard_size = request.form.get('standard_size')

            if selected_size == 'standard':
                if standard_size == 'small':
                    desired_width = 21.138
                    desired_height = 23.243
                elif standard_size == 'medium':
                    desired_width = 35
                    desired_height = 35
                elif standard_size == 'large':
                    desired_width = 23.45
                    desired_height = 5
            elif selected_size == 'custom':
                desired_width = float(request.form['custom_width'])
                desired_height = float(request.form['custom_height'])
                selected_unit = request.form['selected_unit']
                if selected_unit == 'cm':
                    desired_width *= 10
                    desired_height *= 10
                elif selected_unit == 'inches':
                    desired_width *= 25.4
                    desired_height *= 25.4

            estimated_dpi = 300
            pixels_per_mm_width = estimated_dpi / 25.4
            pixels_per_mm_height = estimated_dpi / 25.4

            for file in uploaded_files:
                if file.filename == '':
                    continue

                image = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                height, width = image.shape[:2]
                desired_width_pixels = int(desired_width * pixels_per_mm_width)
                desired_height_pixels = int(desired_height * pixels_per_mm_height)
               
#haarcascade_frontalface_default.xml

                aspect_ratio = width / height
                if aspect_ratio >= 1.0:  # Horizontal image
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces)==0:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    
                    
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 0:
                    errors.append(f"No faces detected in {file.filename}.")
                else:
                    retval, buffer1 = cv2.imencode('.jpg', image)
                    image_base = base64.b64encode(buffer1).decode()
                    
                    uploads.append({
                'name': file.filename,
                'image': f"data:image/jpeg;base64,{image_base}"
                    })
                    (x, y, w, h) = faces[0]
                    top_of_head = int(y - 0.42 * h)
                    new_y = max(0, top_of_head)
                    new_h = h + (y - new_y)
                    roi_x = max(0, x - int(0.2 * w))
                    roi_y = new_y
                    roi_w = min(image.shape[1] - roi_x, int(1.4 * w))
                    roi_h = min(image.shape[0] - roi_y, int(1.4 * new_h))
                    cropped_image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    center = (int(x - roi_x + roi_w // 2), int(y - roi_y + roi_h // 2))
                    if w > h:
                          angle = -90
                    else:
                        angle = 0

                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aligned_image = cv2.warpAffine(cropped_image, rotation_matrix, (roi_w, roi_h))
                    desired_width_pixels = int(desired_width * pixels_per_mm_width)
                    desired_height_pixels = int(desired_height * pixels_per_mm_height)
                    resized_image = cv2.resize(aligned_image, (desired_width_pixels, desired_height_pixels))
                    processed_images.append(resized_image)
                    retval, buffer = cv2.imencode('.jpg', resized_image)
                    image_base64 = base64.b64encode(buffer).decode()
                    resultimage.append(f"data:image/jpeg;base64,{image_base64}")
                    result.append({
                'name': file.filename,
                'image': f"data:image/jpeg;base64,{image_base64}"
                     })
        

        return render_template('usp.html', result=result, errors=errors, original_images=uploads)
@app.route('/clear_arrays', methods=['POST'])
def clear_arrays():
    global processed_images, errors
    processed_images = []
    errors = []
    return redirect(url_for('upload_files'))


@app.route('/download_all')
def download_all_images():
        if not processed_images:
            return "No processed images to download."

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            for idx, (img, img_name) in enumerate(zip(processed_images, input_image_names)):
                img_path = f"{img_name}"  # Use the input image name as the output image path
                cv2.imwrite(img_path, img)
                zipf.write(img_path)
                os.remove(img_path)  # Remove the temporarily saved image
            
        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name="processedimages.zip")
@app.route('/download_errors')
def download_errors():
    error_messages = "\n".join(errors)
    # Create a text file containing the error messages
    response = Response(error_messages, content_type='text/plain')
    response.headers["Content-Disposition"] = "attachment; filename=error_messages.txt"
    return response


if __name__ == '__main__':
        app.run(debug=True)
