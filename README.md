# 😊 Face Emotion Detection using OpenCV

This project detects and classifies human emotions in real-time using a webcam feed. It uses OpenCV for image processing and a pre-trained Convolutional Neural Network (CNN) model to recognize emotions such as Happy, Sad, Angry, Surprise, and more.

## 📂 Project Structure

face_emotion_detection/
├── haarcascade_frontalface_default.xml   # Haar cascade for face detection  
├── model.h5                              # Pre-trained CNN model for emotion classification  
├── labels.pkl                            # Encoded emotion labels  
├── emotion_detect.py                     # Main Python script  
├── README.md                             # Project description and instructions  

## 🚀 Features

- Real-time emotion detection via webcam  
- Uses OpenCV for face detection  
- CNN-based emotion classification  
- Visual display of detected emotion on video feed  

## 🛠️ Requirements

Install the dependencies using:  
pip install -r requirements.txt

Required Libraries:  
- opencv-python  
- keras  
- tensorflow  
- numpy  
- pickle  

## ▶️ How to Run

1. Clone the repository:  
git clone https://github.com/priynsh028/face_emotion_detection.git  
cd face_emotion_detection

2. Run the script:  
python emotion_detect.py

3. Your webcam will start. It will detect faces and display the predicted emotion on the screen.

## 🎯 How It Works

1. The webcam captures a video stream.  
2. Haar Cascade is used to detect faces.  
3. The detected face is preprocessed and passed to a trained CNN model.  
4. The model predicts the emotion.  
5. The predicted emotion is displayed in real-time.  

## 📸 Example Output

A screenshot can be placed here showing face detection and emotion label.

## 🧠 Model Details

- Trained on the FER2013 dataset (https://www.kaggle.com/datasets/msambare/fer2013)  
- Input shape: 48x48 grayscale images  
- Output: One of the predefined emotion classes  

## 📌 Emotion Classes

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

## 🙌 Credits

Developed by Priyansh Vanra  
If you found this project helpful, feel free to ⭐ the repo.

## 📬 Contact

For any feedback or suggestions, connect with me:  
Email: [your-email@example.com]  
Twitter: [@yourhandle]  
LinkedIn: linkedin.com/in/yourprofile

## 📄 License

This project is licensed under the MIT License.
