# Multiclass Animal Classification using MobileNetV2

## Project Overview
This project implements a multiclass image classification model to categorize animals into 90 different classes. The dataset is sourced from Kaggle and trained using the MobileNetV2 architecture on Google Colab with GPU acceleration.

## Dataset
- **Name:** Animal Image Dataset (90 Different Animals)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
- **Number of Classes:** 90
- **Format:** Images organized in folders by class

## Requirements
Before running the code, ensure you have the following dependencies installed:

```bash
pip install kagglehub tensorflow numpy matplotlib
```

## Steps to Run the Project
1. **Download the Dataset**
   ```python
   import kagglehub
   path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
   ```

2. **Verify GPU Availability**
   ```python
   import tensorflow as tf
   physical_devices = tf.config.experimental.list_physical_devices('GPU')
   if len(physical_devices) > 0:
       tf.config.experimental.set_memory_growth(physical_devices[0], True)
       print('GPU is being used.')
   else:
       print('GPU is not being used.')
   ```

3. **Explore the Dataset**
   ```python
   import os
   dataset_path = "<path_to_dataset>/animals/animals"
   classes = os.listdir(dataset_path)
   print(f'Number of Classes: {len(classes)}')
   print(f'Classes: {classes}')
   ```

4. **Visualize Sample Images**
   ```python
   import matplotlib.pyplot as plt
   import random
   plt.figure(figsize=(15,10))
   for i, class_name in enumerate(classes[:10]):
       class_path = os.path.join(dataset_path, class_name)
       img_name = os.listdir(class_path)[0]
       img_path = os.path.join(class_path, img_name)
       img = plt.imread(img_path)
       plt.subplot(2, 5, i+1)
       plt.imshow(img)
       plt.title(f'{class_name}')
       plt.axis('off')
   plt.show()
   ```

5. **Build and Train the Model**
   - Use MobileNetV2 as the base model.
   - Add custom layers for classification.
   - Train using an Adam optimizer with categorical cross-entropy loss.

6. **Evaluate and Test**
   - Compute classification metrics.
   - Visualize accuracy and loss trends.

## Results
The trained model achieves high accuracy in identifying animal species based on image features. Performance can be further improved by fine-tuning hyperparameters or using data augmentation.

## Future Improvements
- Implement data augmentation techniques.
- Experiment with different CNN architectures.
- Optimize model performance using transfer learning techniques.

## Acknowledgments
- Kaggle for providing the dataset.
- Google Colab for free GPU access.
- TensorFlow/Keras for deep learning frameworks.

## License
This project is for educational purposes only.

