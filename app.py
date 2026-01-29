from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import numpy as np, cv2, os
from efficientnet_pytorch import EfficientNet
from gradcam_utils import generate_gradcam

app = Flask(__name__)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
model.load_state_dict(torch.load('models/efficientnet_model.pth', map_location='cpu'))
model.eval()

classes = ['Potassium Deficiency (-K)', 
           'Nitrogen Deficiency (-N)', 
           'Phosphorus Deficiency (-P)', 
           'Fully Nourished (FN)']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file: return "No file uploaded"
    path = os.path.join('static/uploads', file.filename)
    file.save(path)

    image = Image.open(path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()
        conf = torch.softmax(output, dim=1)[0][pred].item()

    cam = generate_gradcam(model, img_tensor, pred)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image.resize((224,224))), 0.6, heatmap, 0.4, 0)
    cam_path = os.path.join('static/results', 'heatmap_'+file.filename)
    cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return render_template('result.html', cls=classes[pred], conf=round(conf*100,2),
                           img_path=path, cam_path=cam_path)

if __name__ == '__main__':
    app.run(debug=True)
