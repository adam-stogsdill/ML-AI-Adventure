4/15/2020

Using the data from https://www.kaggle.com/c/dogs-vs-cats.
This is a basic CNN using Tensorflow 2.

Preprocessing Stage:
	Sorts the dog and cat images. Shrinking them into 300x300 images. 
	Also uses grayscale to avoid using RGB data and simplifiying the model
	due to computer limitations. 
	Creates a label array along side this process.
		0 = Dog
		1 = Cat
	Lastly, we shuffle the data. However, we could probably have avoided this step
	as the data is trained stochastically anyway

Structure of the Model.

Required Input: 3-Dimensional (300x300x1) This algorithms trains stochastically to avoid bias in either label.
Layer 1: 2D-Convolutional Layer, 32 filters, kernel_size = (3x3), stride = 1, padding = same, activation_function = relu
Layer 2: Max Pooling Layer (2x2)
Layer 3: 2D-Convolutional Layer, 32 filters, kernel_size = (3x3), stride = 1, padding = same, activation_function = relu
Layer 4: Flatten Layer
Layer 5: Dropout Layer (40% dropout rate)
Layer 6: Dense Layer (1024 Neurons, activation_function = relu)
Layer 7: Dropout Layer (40% dropout rate)
Layer 8: Dense Layer (512 Neurons, activation_function = relu)
Output Layer: Dense Layer (2, activation_function = softmax)

Model uses the Adam optimization algorithm, (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam),
the Sparse Categorical Entropy loss function, (https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy),
and looking at accuracy as the prime metric.
