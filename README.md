
<h1>BlurSense: A Deep Learning-Based Approach for Motion and Defocus Blur Detection</h1>

<p><b>Overview:</b> BlurSense is an AI-driven system designed to detect whether an image is <i>blurred</i> or <i>sharp</i>. The project tackles the challenge of distinguishing between different types of blur — primarily <b>Defocus (Gaussian or out-of-focus blur)</b> and <b>Motion blur</b> — using both traditional image processing methods and deep learning-based approaches. The solution enhances image quality analysis in applications like photography, surveillance, and medical imaging.</p>

<h2>🎯 Objective</h2>
<p>To accurately classify images as <b>blurred</b> or <b>sharp</b>, while distinguishing between two main types of blur:</p>

<p><b>Defocus / Gaussian Blur:</b> Caused by out-of-focus capture or low contrast.</p>
<p><b>Motion Blur:</b> Caused by camera or object movement during exposure, introducing directional patterns.</p>

<h2>🧠 Technical Approach</h2>

<h3>1. Traditional Computer Vision Approach</h3>
<p><b>Method:</b> Laplacian Variance (OpenCV)</p>
<p><b>Process:</b> Compute the Laplacian of an image to estimate sharpness using intensity variance as a threshold for blur detection.</p>
<p><b>Observation:</b> Achieved ~77% accuracy but struggled to detect motion blur, as directional edges are not captured by Laplacian variance. Sobel filters were also tested but led to increased misclassification.</p>

<h3>2. Deep Learning Approach</h3>
<p><b>Model:</b> Fine-tuned ResNet-18 CNN</p>
<p><b>Rationale:</b> CNNs learn spatial and directional patterns, enabling better detection of motion and defocus blur. ResNet’s residual connections help prevent vanishing gradient issues during training.</p>
<p><b>Training Details:</b> The model was trained on a custom dataset using Google Colab GPU for 5 epochs (~7 hours runtime due to limitations). Despite limited training, the CNN reduced misclassified sharp images from 107 to 59.</p>

<h3>Pipeline Summary:</h3>
<p>• Image preprocessing and normalization</p>
<p>• Laplacian and Sobel variance-based sharpness estimation (baseline)</p>
<p>• CNN fine-tuning using ResNet-18</p>
<p>• Model evaluation using accuracy and confusion matrix metrics</p>

<h2>⚙️ Model Architecture</h2>
<p><b>Base Model:</b> ResNet-18 (Pretrained on ImageNet)</p>
<p><b>Loss Function:</b> Cross-Entropy Loss</p>
<p><b>Optimizer:</b> Adam</p>
<p><b>Learning Rate:</b> 0.001</p>
<p><b>Batch Size:</b> 32</p>
<p><b>Frameworks Used:</b> PyTorch, OpenCV, NumPy, Matplotlib</p>

<h2>📊 Dataset Description</h2>
<p>The dataset consists of labeled images categorized as:</p>
<p>• <b>Sharp Images</b></p>
<p>• <b>Defocus / Out-of-Focus Blur</b></p>
<p>• <b>Motion Blur</b></p>

<p>Each image was resized, normalized, and split into training and testing subsets. The dataset also includes diverse lighting and texture conditions to enhance generalization.</p>

<h2>📈 Results and Analysis</h2>

<table>
<tr><th>Method</th><th>Accuracy</th><th>Misclassified Sharp Images</th><th>Observation</th></tr>
<tr><td>Laplacian Variance</td><td>~77%</td><td>107</td><td>Failed to capture motion blur edges</td></tr>
<tr><td>Sobel Filter</td><td>~70%</td><td>120</td><td>Increased false positives for blurry images</td></tr>
<tr><td>ResNet-18 (Fine-Tuned)</td><td><b>~90%+</b></td><td><b>59</b></td><td>Accurately differentiates both blur types</td></tr>
</table>

<p><b>Key Insight:</b> The CNN outperforms traditional methods by learning texture and direction-based blur characteristics.</p>

<h2>🚀 Features</h2>
<p>✅ Detects both motion and defocus blur</p>
<p>✅ Compares traditional and deep learning methods</p>
<p>✅ Fine-tuned ResNet-18 with improved classification accuracy</p>
<p>✅ Laplacian variance baseline implementation</p>
<p>✅ Handles low-contrast and directionally blurred images</p>
<p>✅ Scalable for real-time applications</p>



<h2>🧪 Output Example</h2>
<pre><code>{
  "image_name": "sample_123.jpg",
  "predicted_label": "Defocus Blur",
  "confidence": 0.91,
  "method_used": "ResNet-18",
  "details": {
    "laplacian_variance_score": 58.32,
    "predicted_category": "Blur",
    "blur_type": "Motion",
    "true_label": "Blur"
  }
}
</code></pre>

<h2>🔮 Future Enhancements</h2>
<p><b>Model Improvements:</b></p>
<p>• Train with larger datasets and higher epoch counts</p>
<p>• Use transfer learning with advanced architectures (DenseNet, Vision Transformers)</p>
<p>• Introduce attention mechanisms for spatial feature focus</p>

<p><b>Application Enhancements:</b></p>
<p>• Develop a real-time blur detection web app</p>
<p>• Support video frame-wise blur detection</p>
<p>• Integrate with image restoration/deblurring pipelines</p>

<h2>📦 Dependencies</h2>
<p><b>torch</b> — Deep learning framework for model training</p>
<p><b>torchvision</b> — Pretrained models and transforms</p>
<p><b>opencv-python</b> — Image preprocessing and Laplacian variance</p>
<p><b>numpy</b> — Numerical computations</p>
<p><b>matplotlib</b> — Visualization and metrics plotting</p>

<h2>🧑‍💻 Author</h2>
<p><b>Name:</b> Tanishka Kasal</p>
<p><b>Institution:</b> JK Lakshmipat University, Jaipur</p>

<p><b>Domain:</b> Deep Learning and Computer Vision</p>

<hr>
<p><b>Note:</b> BlurSense was developed to explore the comparative performance of traditional vision-based blur detection and deep learning methods. The model demonstrates significant improvements in identifying complex blur types, showing potential for real-world deployment in visual quality analysis systems.</p>


