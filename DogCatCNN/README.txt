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