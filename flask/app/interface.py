from flask import Flask, redirect, render_template, request, session, url_for, send_file, Response
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

import os
import io
import base64
import matplotlib.pyplot as plt
from boto3 import client
from . import app
from .tf2.models import Model
from .tf2.generate import generate 
from .query import Validate
from . import aws_id, aws_key

dropzone = Dropzone(app)


app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

model=None
input_image = None
@app.route('/', methods=['GET', 'POST'])
def index():
    global model
    global input_image
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    else:
        if len(session['file_urls'])>=2:
            session['file_urls']= []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']
    # handle image upload from Dropzone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename    
            )
            # append image urls
            file_urls.append(photos.url(filename))
        session['file_urls'] = file_urls
        input_image = os.getcwd() + '/uploads/' + filename
        if (len(session['file_urls'])==2) and (model is None):
            model = Model(input_image)
            model.model_load('app/tf2/tf2vae/checkpoints/weight_10.h5')
        return "uploading..."
    
    # return dropzone template on GET request    
    return render_template('index.html')


@app.route('/results')
def results():
    global model
    global input_image
    img_recon = None
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
    img_in, img_gen= model.VAE(input_image)
    img_recon = generate(img_in, img_gen)
    img = (img_recon * 255).astype('uint8')
    candidates = model.NN_similar(img_recon)
    # query
    validate = Validate(candidates[0:4])
    table, uiid_list = validate.fetch()
    # download image from s3
    # create file-object in memory
    s3 = client('s3', aws_access_key_id = aws_id, aws_secret_access_key = aws_key)
    Bucket="hxzhuang"
    imgs = []
    for i in range(len(uiid_list)):
        f_obj = io.BytesIO()
        s3.download_fileobj(Bucket,'rico/unique_uis/{}.jpg'.format(uiid_list[i]), f_obj)
        f_obj.seek(0)
        img_s3 = base64.b64encode(f_obj.getvalue()).decode()
        imgs.append(img_s3)
    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    plt.imsave(file_object, img)
    file_object.seek(0)
    result = base64.b64encode(file_object.getvalue()).decode()
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
#     session.pop('file_urls', None)
    
    return render_template('results.html', file_urls=file_urls, result = result, img0 = imgs[0], img1= imgs[1], img2=imgs[2], img3=imgs[3],table=table)
