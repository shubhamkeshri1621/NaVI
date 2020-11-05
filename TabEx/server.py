from flask import Flask, render_template, request ,flash
from werkzeug.utils import secure_filename
import tabx
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PIL import Image
import pytesseract
import io
UPLOAD_FOLDER = '/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload')
def upload():
   return render_template('/index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():

   if request.method == 'POST':
      file = request.files['file']
      in_memory_file = io.BytesIO()
      if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
      file.save(in_memory_file)
      data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
      
      img = cv2.imdecode(data, 0)  
      print("[*]processing...")    
      table=tabx.main(img)
      
      # return table.render()
      return render_template("imdex.html", name=table.render())
		
if __name__ == '__main__':
   app.run(debug = True)
