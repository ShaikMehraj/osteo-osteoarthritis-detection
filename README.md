
In this project, we will developed a deep learning model to detect osteoarthritis in patients. We will do this by training the model on two datasets of x-rays: one dataset of x-rays from patients with osteoarthritis and another dataset of
x-rays from patients without osteoarthritis. It is often difficult to detect osteoarthritis with the naked eye, especially in its early stages. This is where deep learning can be helpful. Deep learning models can be trained to identify 
patterns in images that are invisible to the human eye . 
We will use a convolutional neural network (CNN) to train our model. CNNs are a type of deep learning model that are well-suited for image classification tasks. Our deep learning model will be trained on a large dataset of x-rays. 
The model will learn to identify the features of osteoarthritis, such as bone spurs, joint space narrowing, and cartilage loss. Once the model is trained, it will be able to classify new x-rays as either 
having osteoarthritis or not.
The proposed system of the project are as follows: 
• EDA and Data Augmentation  
• CNN model  
• Compile and train  
• Visualization and evaluate 
HARDWARE REQUIREMENTS 
• Processor: Intel or AMD  
• RAM: 16 GB 
• Space on disk: minimum 1000 MB 
• For running the application on Device: laptop with GPU is preferred 
• Minimum space to execute: 16gb ram with minimum of 8 core  
• GPU: NVidia graphic card.  
SOFTWARE REQUIREMENTS 
• Operating System: Any OS  

• The following NVIDIA® software are only required for GPU support.
    ○ [NVIDIA® GPU drivers version 450.80.02 or higher](https://www.nvidia.com/download/index.aspx).
    ○ CUDA® Toolkit 11.8.
    ○ cuDNN SDK 8.6.0.
• Network: Wi-Fi Internet or cellular Network.  
• Jupyter Notebook or Google Collab: Used to run our dataset, compute require results, and create models and execute them on the dataset and to obtain the accuracies.  
• Packages: All TensorFlow and keras files and data visualization libraries. 
DATASET:  
"kl-0," "kl-3," and "kl-4": These datasets are presumably labelled with the Kellgren Lawrence (KL) grades, which are often used to assess the severity of knee osteoarthritis. 
The KL grading system typically ranges from 0 to 4, with 0 indicating a normal knee and 4 indicating severe osteoarthritis.
The use of the Kaggle knee osteoarthritis dataset for testing and validation ensures that your model is evaluated on a dataset that is representative of real-world data.
ResNet152v2 model: 
ResNet152v2 is a deep learning model that is based on the convolutional neural network (CNN) architecture. CNNs are a type of neural network that are well-suited for image classification tasks. ResNet152v2 is a particularly powerful CNN 
architecture, as it is able to learn complex features from images.
 classification of osteoarthritic and normal knee images using ResNet152V3 as a model and KL-0 and KL-3,4 as training data. The proposed model achieved a validation accuracy of 95.69% and a test accuracy of 91.36%. 
 The model also achieved high precision and recall for both classes.
