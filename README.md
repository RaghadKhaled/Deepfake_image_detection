# Deepfake image detection on faces using CNN and SVM

<div align="center">
<p>
<img src="ezgif.com-gif-maker (1).gif" width="700" height="400"/>
</p>
<br>
<div>
</div>
</div>


---
This is a college project required for a Machine Learning course.<br>
It is done as teamwork by Alhanouf Almansour, Khloud Alnufaie, Raghad Albosais and Weaam Alghaith.

---
## Table of content
* [Project overview](#project-overview)
* [Data description](#data-description)
* [Model architecture](#model-architecture)
* [Tools](#tools)
* [How to run](#how-to-run)
* [Contributors](#contributors)
---

### Project overview

One of the strong techniques used in creating misinformation has become known recently as "Deepfake". Deepfakes increasingly threaten the privacy of individuals. Furthermore, Deepfakes can distort our perception of the truth and deceive us. The content of an image can shake the world either because it sparks controversy, or discredits someone. An individual may be accused or suspected of a situation that did not actually occur. For example, modifying a person’s expression to appear sad when in reality, they were happy to satisfy a fake narrative. In response, We have used transfer learning strategy to extract features from images, through using pre-trained CNN model, which is VGG16 combined with SVM classifier that is trained using the extracted features provided by VGG16. To illustrate the functionality in the model, we have included production by deployment model.

---

### Data description

We Detect fake image using [Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) dataset from Kaggle.
The dataset contains expert-generated high-quality photoshopped face images. 
The images are composite of different faces, separated by eyes, nose, mouth, or whole face. 
The full dataset contains 452 MB of faces Inside the parent directory, training_real/training_fake contains real/fake face photos, respectively. 
In case of fake photos, they have three groups: easy, mid, and hard. We plan to use 500 images with ground truth, split them as 75% training, and 25% test to evaluate models using this.


---

### Model architecture

![Model architecture](https://user-images.githubusercontent.com/68460588/166117329-f3a011bc-9a74-479f-81bc-1d8247d12af7.jpg)

The image is preprocessed using three image preprocessing techniques (image resize, color space conversion, and normalization). 
Next, the image is moved into a pre-trained convolutional neural network (CNN) feature extractor, called VGG16 which extract spatial feature from the image.  
After that, the output of the feature extractor is fed as input to the support vector machine (SVM) classifier.

---

### Tools

- Libraries: 

![Pandas](https://img.shields.io/badge/pandas-330F63??style=flat&logo=pandas&logoColor=white)
![openCV](https://img.shields.io/badge/openCV-%23F7931E.svg??style=flat&logo=openCV&logoColor=black&color=9cf)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg??style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg??style=flat&logo=scikit-learn&logoColor=white)
![keras](https://img.shields.io/badge/keras-%23000.svg??style=flat&logo=keras&logoColor=white&color=red)
![tensorflow](https://img.shields.io/badge/tensorflow-%23000.svg??style=flat&logo=tensorflow&logoColor=white&color=green)
![seaborn](https://img.shields.io/badge/seaborn-%2006600.svg??style=flat&color=blue)
![Flask](https://img.shields.io/badge/Flask-%233F4F75.svg??style=flat&logo=flask&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-%233F4F75.svg??style=flat&&logo=matplotlib&color=yellow)


- Softwares: 

![Google colab](https://img.shields.io/badge/Googlel%20Colab-0078d7.svg??style=flat&logo=google-colab&logoColor=orang)
![PyCharm](https://img.shields.io/badge/PyCharm-%233F4F75.svg??style=flat&logo=pycharm&logoColor=white)

---

### How to run
-	Download (Deepfake detector interface) folder that contain required files to run the interface. 
-	Unzipe 'Deepfake detector' folder in the 'Deepfake detector interface' folder.
-	Put the content of unizped folder (Deepfake detector.pkl) in the 'Deepfake detector interface' folder.
-	Run app.py file (in Deepfake detector interface folder) using any Python IED. 
-	Open the link of the interface and upload image from your device to classify it as fake or real.


---

### Contributors

[@AlhanoufAlmans](https://github.com/AlhanoufAlmans)

[@hkh7897](https://github.com/hkh7897)

[@RaghadKhaled](https://github.com/RaghadKhaled)

[@Weaam20](https://github.com/Weaam20)
