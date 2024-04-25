# A Comparative Study of CIFAR-10 Classification using PCA-HOG-SVM and CNN

This repository contains the code and results of a comparative study on CIFAR-10 image classification using two different approaches: PCA-HOG-SVM and Convolutional Neural Networks (CNN).

## Table of Contents
1. Introduction
2. PCA-HOG-SVM
3. Pipelining
4. Convolutional Neural Networks
5. Results
6. Conclusion

## Introduction
![image](https://github.com/javarath/CIFAR-10-PCA-HOG-SVM-vs-CNN/assets/102171533/3244c903-2ef3-4829-a911-7e9b043458cc)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. In this study, we compare the performance of two different machine learning approaches on this dataset. The dataset consists of 60,000 color images. Each image is 32x32 pixels in size. The images are divided into 10 different classes, with each class representing a different type of object. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each class contains exactly 6,000 images. The dataset is split into a training set and a test set. The training set contains 50,000 images, while the test set contains 10,000 images. This allows machine learning models to be trained on a large amount of data, and then tested on separate data to evaluate their performance. The dataset is divided into five training batches and one test batch, each with 10,000 images. 

The CIFAR-10 dataset is publicly available and can be used for non-commercial research purposes It’s a popular choice for researchers due to its size and diversity, making it suitable for a wide range of experiments in image classification. It is also used to evaluate the performance of various machine learning models, including convolutional neural networks (CNNs), which have achieved state-of-theart results on this dataset.

## PCA-HOG-SVM
# PCA
PCA is a statistical technique used in machine learning and data analysis. It's primarily used for dimensionality reduction, transforming a large set of variables into a smaller one. The transformed variables, called principal components, are uncorrelated and retain most of the information from the original dataset. PCA works by identifying the directions (principal components) in which the data varies the most and reorienting the dataset along these directions. This makes it easier to visualize and analyze high-dimensional data. It's widely used in exploratory data analysis and for making predictive models.

# Histogram of Oriented Gradients (HOG):
![image](https://github.com/javarath/CIFAR-10-PCA-HOG-SVM-vs-CNN/assets/102171533/ccbfccf3-e6eb-40f6-9c08-42da91cf8d40)
Reference: https://customers.pyimagesearch.com/lesson-sample-histogram-of-oriented-gradients-and-car-logo-recognition/


The Histogram of Oriented Gradients (HOG) is a feature descriptor used in computer vision for object detection. It works by computing the gradient values of pixels, casting votes for histogram channels based on these gradients, grouping these into larger blocks, and normalizing these blocks to account for changes in illumination and contrast. The HOG descriptor is typically used as input to a classifier like a Support Vector Machine (SVM) to predict if an image contains the object of interest. It's effective for tasks like object detection and image recognition and is used to evaluate the performance of various machine learning models, including Convolutional Neural Networks (CNNs). In the context of the CIFAR-10 dataset, the classes are mutually exclusive with no overlap between automobiles and trucks.

# Support Vector Machines (SVMs):
Sure, here's a brief explanation:

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It's particularly effective when the number of dimensions is greater than the number of samples. 

1. **Hyperplane**: SVM works by finding an optimal hyperplane that separates the data points of different classes in the feature space. The dimension of the hyperplane depends on the number of features.

2. **Support Vectors**: These are the data points that lie closest to the decision surface (or hyperplane). They are the most difficult to classify and influence the location of the decision surface.

3. **Margin**: SVMs aim to maximize the margin around the separating hyperplane. A larger margin helps in reducing the generalization error of the model.

4. **Kernel Trick**: For data that is not linearly separable, SVM uses the kernel trick. This involves transforming the input data into a higher-dimensional space where a hyperplane can be used to separate the data.

5. **Robustness**: SVMs are robust to outliers and can handle them effectively. They find the hyperplane that maximizes the margin, so even if an outlier is present, SVMs will choose a hyperplane that has the maximum distance to the nearest data points from each class.

SVMs are good for CIFAR-10 because they can handle high-dimensional data and construct a hyperplane in a high-dimensional space that can classify the images effectively. The kernel trick is particularly useful for the CIFAR-10 dataset as the data is not linearly separable. Transforming the data to a higher dimension where it becomes separable can lead to better classification results. In conclusion, SVMs are a good choice for tasks like image classification on the CIFAR-10 dataset due to their ability to handle high-dimensional data, robustness to outliers, and use of the kernel trick.

## Pipelining

Pipelining is a technique that allows to chain multiple steps of a machine learning workflow into a single object. This simplifies the code and avoids intermediate variables. In this project, a pipeline is created using sklearn.pipeline.Pipeline() that consists of three steps: StandardScaler, PCA, and SVC. The pipeline can be fitted to the training data and used to make predictions or find the score on the test data.

# Results:
The overall accuracy of the model is 0.63. The macro average and weighted average for precision, recall, and F1-score are also 0.63. This indicates that the model has a balanced performance across all classes. However, there is room for improvement, especially for classes with lower F1-scores. Further tuning of the model or use of more complex models may lead to better results.

![image](https://github.com/javarath/CIFAR-10-PCA-HOG-SVM-vs-CNN/assets/102171533/4cf9c6ce-1722-47b3-93c2-9b11121af712)

![image](https://github.com/javarath/CIFAR-10-PCA-HOG-SVM-vs-CNN/assets/102171533/64f97d99-ed45-48aa-a8fa-9dcb3f405e1e)

## Convolutional Neural Networks
- **CNNs** are a type of **Deep Learning** neural network architecture primarily used in **Computer Vision** tasks.
- They are designed to process data with a **grid-like topology**, such as an image composed of pixels.
- The architecture of a CNN is designed to leverage the **2D structure** of an input image. This is achieved through **local connections** and **tied weights**, followed by some form of **pooling**. This results in **translation invariant features**, meaning the network can recognize patterns regardless of their position in the image.
- CNNs have their neurons arranged in **three dimensions**: width, height, and depth. The neurons in one layer do not connect to all the neurons in the next layer but only to a small region of it.
- The final output of a CNN is a single vector of **probability scores**, organized along the depth dimension.
- CNNs are highly efficient in **image classification tasks**. They learn hierarchical representations of the data, with lower layers detecting simple patterns and deeper layers recognizing more complex structures.
- In the context of the **CIFAR-10 dataset**, which consists of 60,000 32x32 color images in 10 classes, CNNs have achieved high accuracy, with test accuracy reaching up to **81%**. This is due to CNNs' ability to automatically learn and extract features from images. 

# Results:
![image](https://github.com/javarath/CIFAR-10-PCA-HOG-SVM-vs-CNN/assets/102171533/e3e0149c-7c77-43d1-bccc-57e53338e4d5)

![image](https://github.com/javarath/CIFAR-10-PCA-HOG-SVM-vs-CNN/assets/102171533/9cc3a976-87c7-439f-8a25-2a5002c2e1e4)

## COMPARISION
![image](https://github.com/javarath/CIFAR-10-PCA-HOG-SVM-vs-CNN/assets/102171533/8015eaf2-7af6-4b7d-96a8-8c7624b2034a)


## Conclusion
In conclusion, both Convolutional Neural Networks (CNNs) and PCA-HOG-SVM have their strengths and weaknesses when applied to the CIFAR-10 dataset.

CNNs, with their ability to automatically learn hierarchical representations of data, have shown impressive performance on the CIFAR-10 dataset. They have achieved high accuracy, precision, recall, and F1-scores across various classes. However, this comes at the cost of high computational power and longer training times due to their complex architecture.

On the other hand, PCA-HOG-SVM, while not as accurate as CNNs, can be a robust option when computational resources or training time are limited. Their simpler architecture leads to faster training times and lower computational requirements. However, the performance of PCA-HOG-SVM heavily depends on the quality of handcrafted features, which can be a limitation.

Therefore, the choice between CNN and PCA-HOG-SVM for CIFAR-10 or any other task should be guided by the specific requirements of the task, including the available computational resources, the amount of available data, and the acceptable trade-off between accuracy and computational efficiency. 


