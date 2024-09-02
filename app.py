from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import torch
import numpy as np
import io
from prediction import pred_class

app = Flask(__name__)

# Load Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('ghostnet_checkpoint_fold1.pt', map_location=device)
model.half()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get file from the form
        file = request.files['image']
        if file:
            # Read image
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            
            # Prediction
            class_name = ['1 Normal', '2 Mild', '3 Severe']
            probli = pred_class(model, image, class_name)
            max_index = np.argmax(probli[0])

            # Prepare results for rendering
            results = {
                'image': file.filename,
                'predictions': [
                    {
                        'class_name': class_name[i],
                        'probability': probli[0][i] * 100,
                        'highlight': i == max_index
                    } for i in range(len(class_name))
                ]
            }
            return render_template('results.html', results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
