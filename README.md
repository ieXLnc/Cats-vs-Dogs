# Deploying a deep learning cats vs dogs classifier web application

You can find the web application [here](https://catsvsdogs.herokuapp.com/)

<center><img src="https://user-images.githubusercontent.com/63811972/182856003-a47fff6e-35fa-4311-b60d-c36e88cda163.gif" height="500" width="800"></center>

## Goals of the project

- Build a __Convolutional neural network__ using TensorFlow to classify between cats and dogs. 
- Build a Flask application containing the model to allow for __real time predictions__.
- Deploy the application using __Docker__ and __Heroku__.


## Steps taken:

### 1. Building the CNN model:

__Find the dataset:__ <br>
The dataset used to create our model comes from the Kaggle competition [Dogs VS Cats](https://www.kaggle.com/c/dogs-vs-cats/data) and is composed of 25.000 labelled cat/dog images.


__Import and preprocess the data:__ <br>
The images were preprocessed with the [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) object of TensorFlow which generates tensor image data batches with real-time data augmentation. To use it, the files have to be compartmentalized in different directories such as:
  <center><img src="https://user-images.githubusercontent.com/63811972/182862131-e4ede621-1ea4-4350-b95e-467c93e635ba.png" height="200" width="250"></center>


__Create the model with MobilNetV2 and adding layers:__ <br>
I used the pre-trained MobileNetV2 model from Tensorflow.keras. MobileNetV2 is a pretrained model that can be used as a base for many visual recognition tasks and that will allow me to save time from building a very complex convolutional neural network architecture and greatly improve our accuracy.

```
mobilenet_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
mobilenet_model.trainable=False
model = tf.keras.Sequential(
    [
    mobilenet_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Flatten(),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
]
)
```

__Accuracy score of our model:__ <br>
Our model achieved a validation accuracy of 98.187% and a testing accuracy of 98.4 % in the 2500 testing images.


### 2. Creating the Flask application 

- ### a. Backend

  - Learn to use Flask: I chose flask for its ease of use to develop small applications that can be easily deployed, [flask documentation](https://flask.palletsprojects.com/en/2.2.x/).
  - Building the Flask application
  - Make predictions in real time:  get the image through the post method of flask, preprocess it and load the model to make our prediction. After the prediction is made, the label and probabilities given bu the model are returned and linked to the HTML page 'classification'.

- ### b. Frontend

  - Connect the HTML code with the backend
  - Make a simple HTML page to upload/classify

### 3. Dockerize our application

- Create the Docker image
- Push my image to Docker hub: the repo is available to clone and use at iexlnc/catvsdog_flaskapp
- Necessary to deploy my image in Heroku

### 4. Deploying our application with Heroku

- Create an account and deploy my app @[Heroku](https://www.heroku.com/platform)


### 5. Result

#### Home page:
<img src="https://user-images.githubusercontent.com/63811972/182874380-ce77e809-1121-4300-9d0a-ba383065a5a0.png">

#### Classification page:
![classify_page](https://user-images.githubusercontent.com/63811972/182874592-e79ff919-b2c3-4cf3-9e0d-375c5338b4e5.png)




