# Steering-Angle-Prediction
A pytorch implementation of the Nvidia paper with some changes.Here we will be predicting the steering angle of a car using the image taken from the dashcam.

### IMPORTANT
Under any circumstances, this code should not be used as software for your own self-driving car.

### How to run the code
1. Download the dataset from [link](https://github.com/SullyChen/driving-datasets).Here i used Dataset 1 of nearly 2.2GB.
2. models contain the trained model using pytorch.
3. src contains the code
   1. config.py --> here you should give the paths used in the program.
   2. convert.py --> This will convert the data.txt into csv file along with conversion of targets into radians(we use radians because it will give small range for predicting than when using degree)
   3. dataset.py --> This will create a dataset with images used for the training the model
   3. engine.py --> This will contain the training code and saving the models
   4. evaluate.py --> After training the model we can visualize the output(try this.)
   5. model.py --> Here we define our pytorch model
   6.train.py --> This will train out model

### Reference
Thankyou [SullyChen](https://github.com/SullyChen) for an amazing dataset.
