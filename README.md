## Code
* [Dockerfile](Dockerfile) a set of commands for creating a docker image
* [requirements.txt](requirements.txt) environment settings
* [server.py](server.py) contains API logic
* [train.py](train.py) downloads the dataset and trains the model based on it

## How to run
* Install and run Docker
* Have at least 10 GB of RAM
* Build Docker image using `docker build . -t news_check`
* Run Docker container using `docker run --rm -it -p 8000:8000 news_check`
* Go to `http://127.0.0.1:8000` or `http://127.0.0.1:8000/docs` to see all available methods of the API

