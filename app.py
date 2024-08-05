from flask import Flask, render_template
import gradio as gr
import requests

app = Flask(__name__)

# Define the Gradio interface
def predict(image):
    HF_API_URL = 'https://api-inference.huggingface.co/models/Sakibrumu/Food_Image_Classification'
    HF_API_TOKEN = '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'
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(HF_API_URL, files={"file": image}, headers=headers)
    return response.json()

iface = gr.Interface(fn=predict, inputs="image", outputs="label")

# Create a Flask route to serve the Gradio interface
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gradio')
def gradio():
    return iface.launch(share=True)

if __name__ == '__main__':
    app.run(debug=True)
