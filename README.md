# Multi-Class-Flower-Classification
Multiple Class Flower Image Classification using Keras

USAGE
For training model : python3 training_model --dataset training_set --model trained_model --plot plot
For predicting image : python3 predict.py --dataset training_set --model trained_model --image test_set/rose1

The model have been trained using Keras Library.

The architecture of the neural network used here is commonly known as LeNet architecture which is depiced as below
INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC

For training this model, the most optimized No of Epochs were 25 and the Batch Size of 32 with the Adam Optimizer with the Initial Learning Rate of 1e-3.

The pre-trained model here has the following accuracy/loss which is also shown in the figure plot.png
Training Accuracy - 0.9057
Validation Accuracy - 0.8864
Training Loss - 0.2225
Validation Loss - 0.2838
