# Welcome to serve_api
created by Poyraz Guler


serve_api is a tool created to load models in a remote machine and create api with endpoints for users to interact through termial.

This program gives you the option to quantized your models before using. keep in mind that you can use already quantized models. If the model has 4bit or 8bit in the title, the program will automatically assume that the model is quantized and will load it as is without checking the extra quantification parameters provided in the .yaml file. 

1. Install required dependencies by running the fallowing code on server terminal.
    """bash
   pip install -r requirements.txt
   """

2. First upload the config.yaml file into the remote server in the same directory as serve-api.py

3. Alter the config.yaml file to change repo_id with the name of your model thats already installed in your server.

4. Inside the config.yaml file set the temperature, max_new_token, quantization settings and torch_dtype to your liking. (bfloat32 is not recommended for 8B or higher models while using NVIDIA GeForce RTX 3060 or a lower GPU)

5. Run the code to create an API
    """bash
   python serve_api.py
   """

6. Use these endpoints to interact with the API:
    ## GET

    1. for checking model name and load status.
    """bash
   curl -X GET "http://127.0.0.1:1420/model_status/"
   """
    2. for checking model size
   """bash
   curl -X GET "http://127.0.0.1:1420/model_size/"
   """
    3. for checking model precision
   """bash
   curl -X GET "http://127.0.0.1:1420/model_precision/"
   """
    4. for checking if the api is running
   """bash
   curl -X GET "http://127.0.0.1:1420/"
   """
    5. for checking to see if there is an available GPU to use
   """bash
   curl -X GET "http://127.0.0.1:1420/gpu/"
   """
   ## POST
   1. to send a prompt and receive a response from the model
   """bash
   curl -X POST "http://127.0.0.1:1420/generate/" -H "Content-Type: application/json" -d '{"prompt": "your prompt here"}'
   """
   2. to reload a new model after changing config.yaml
   """bash
   curl -X POST "http://127.0.0.1:1420/reload_config/"
   """
    
# further improvements and notes:

The program has been tested with 3 generic models. 
    "core-outline/gemma-2b-instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Yi-1.5-6B-bnb-4bit"
The program has not been tested with custom models or fine-tuned models. Issues could arise.

The program has a model class that has unload_model() function never utilized. This function is for future updates where we could create a terminal interface that lets you navigate through models without closing the api. 

For now, the program creates a Url for the api that can only be reached locally. The program could create an api url that can be reached from outside the server.

The endpoint provided for the output generation utilizes only one prompt. Could be tailored for system and user prompts depending on what the endpoints use is.

The output is displayed in a basic print function with no formatting. This could be improved for human readability.


## Dockerization
To dockerize the code, we have written a dockerfile. First you need to create your docker image and then you will have to configure the docker-compose file to your specifications before creating and deploying the docker container.

Keep in mind that we are using nvidia gpu's, So to properly run the code on gpu we need to have  NVIDIA Container Toolkit on our machine. here is how you can download it:

"""bash 
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
"""
For complications, here is the documentation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Here is a guide on how to configure docker-compose
1. Declare the image name you want to create:
    image: <"your_image">
2. The Container listens on your local port 1420. exports it to 1420.
    ports:
      - "1420:1420"
3.  Mount your model files from your machine to the container to avoid downloading them again. Mount your config.yaml file to be able to alter it for reloading.
   where models are downloaded by default:
    ~/.cache/huggingface/hub

    volumes:
      - ~/.cache/huggingface/hub:/root/.cache/huggingface/hub

      - ./config.yaml:/app/config.yaml


Navigate to the folder where you have your dockerfile and run:

to create an image:
"""bash
docker-compose build
"""

to run the container:
"""bash
docker-compose up
"""

to do both simultaneously:
"""bash
        docker-compose up --build 
"""

Now you can alter your config.yaml file and the application running on container will be able to listen to the changes after reloading config. 

You can interact with the api same as with the non-dockerized version.
