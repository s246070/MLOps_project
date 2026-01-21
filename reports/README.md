## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

9

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s246103, s246215, s246070, s245414, s240745

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We did not use any open-source packages outside the course-provided tools and libraries. All dependencies we relied on, including FastAPI, scikit-learn, Hydra, Google Cloud SDK packages, Weights & Biases, and profiling or visualization tools like TensorBoard, SnakeViz, and memory-profiler, were part of the curriculum.
All dependencies are documented in our ```bash pyproject.toml ```.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used ```bash uv``` to manage all Python dependencies in our project. Instead of using ```bash requirements.txt```, we defined our dependencies in a ```bash pyproject.toml``` file, grouped by core, dev, and optional packages. The exact versions were locked using a ```bash uv.lock``` file, ensuring full reproducibility across environments.
To set up a complete and isolated development environment, a new team member would only need to:
```bash uv venv``` - creates a virtual environment
```bash uv sync``` - installs all locked dependencies
This setup is fast, deterministic, and avoids dependency conflicts. It ensures everyone on the team runs with exactly the same versions of all tools and libraries, including testing and documentation tools.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized our project using the cookiecutter template. We followed the suggested structure closely but made a few practical adjustments.

We used the ```bash src/mlops_project``` directory for all source code, including training logic, model loading, and evaluation. The HTTP-based API used for inference is implemented in a subfolder ```bash src/mlops_project/api/```. This separation made it easier to deploy the model independently from the training pipeline.

We did not use the ```bash docs/``` or ```bash notebooks/``` folders. Instead of ```bash requirements.txt```, we managed dependencies entirely through ```bash pyproject.toml``` using the ```bash uv``` dependency manager.

We added a ```bash wandb/``` folder to store experiment tracking logs from Weights & Biases, and a ```bash profiling/``` folder containing ```bash profile_training.py``` to analyze training performance using tools like SnakeViz and memory-profiler. These folders were not part of the original template.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used ```bash ruff``` for both linting and formatting, which was enforced via GitHub Actions in two workflows: ```bash ci.yml``` and ```bash ruff_autocheck.yml```. The CI workflow automatically checks code style, formatting, and runs our tests on every push or pull request. The ```bash ruff_autocheck.yml``` workflow runs ```bas ruff``` in auto-fix mode on selected branches, committing fixes when possible.

We also used Python’s built-in type annotations to improve code readability and catch type-related issues early, though we did not enforce type checking with a separate tool.

Code style and formatting rules matter in larger projects because they reduce friction in collaboration, prevent merge conflicts, and make the codebase more maintainable. Linting helps catch bugs or bad practices before they become problems, and formatting ensures consistency regardless of who wrote the code.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total, we implemented 7 unit and integration tests, along with a Locust-based load test. Our unit tests cover key components of the codebase, including the dataset loading, model architecture, and training logic. For instance, we test model initialization, output shapes, and error handling when input dimensions are incorrect.

We also implemented integration tests for the API, verifying both valid and invalid inference requests as well as the health check endpoint. Finally, our Locust load test simulates concurrent users hitting the prediction and health endpoints to measure API performance under load.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- question 8 fill here ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made use of both branches and pull requests (PRs) throughout our project. Each group member typically created a feature or fix branch when working on new functionality. This allowed us to develop and test code in isolation without directly affecting the ```bash main``` branch.

When a task was completed, a pull request was opened to merge the branch into ```bash main```. These PRs had to be reviewed and approved by two team members, which helped catch bugs early, enforce consistent code style, and encourage collaboration through feedback.

Using branches and PRs improved our version control by making the development process safer and more transparent. It ensured that code was peer-reviewed before becoming part of the production-ready codebase, and gave us a clear history of changes and who made them.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did not use DVC in our project, but we implemented data version control using a different approach.
We created a Google Cloud Storage (GCS) bucket to store and version our trained models. In our application code, we used the ```bash google-cloud-storage``` package to programmatically load the correct model version from the bucket using environment variables (```bash MODEL_BUCKET``` and ```bash MODEL_BLOB```). This allowed us to separate model storage from the application logic and easily switch between model versions by changing the path to the model file in the bucket.
Although DVC provides more automation and tracking, our simpler setup was sufficient for our use case and gave us flexibility. In larger projects, however, DVC would be beneficial to formally track changes in data and models across experiments and ensure full reproducibility.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have organized our continuous integration into four separate GitHub Actions workflows. The main workflow, ```bash ci.yml```, runs on all pushes and pull requests. It handles unit testing using ```bash pytest```, code linting using ```bash ruff```, and formatting checks to ensure that all code contributions follow our quality standards. Additionally, we have an auto-fix workflow, ```bash ruff_autocheck.yml```, which runs on the main and tests branches. This workflow uses ```bash ruff``` to automatically fix safe linting and formatting issues and commits them back to the repository if changes are made.

To validate that our deployed API is functioning correctly, we created a workflow file called ```bash test_api.yml```, which sends test requests to our live Cloud Run deployment and checks the responses. We also implemented ```bash loadtest.yml```, a workflow for load testing our API using Locust. This simulates multiple users interacting with the deployed API and uploads the performance results for inspection.

This continuous integration setup helped us maintain consistent code quality, identify bugs early, and verify that our API remained reliable and responsive after each deployment.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured our experiments using Hydra configuration files. Our main ```bash config.yaml``` file defines the overall setup, including model type, dataset paths, training hyperparameters, and logging. This base config can be overridden using experiment-specific files, such as ```bash exp1.yaml```, which sets a different random seed and custom WandB project name.

We stored hyperparameters in a separate ```bash hyperparameters/``` folder, allowing for easy overrides when launching different experiments. For example, to run a specific experiment with our ```bash exp1.yaml```, we could run:

```bash python src/mlops_project/train.py --config-path experiments --config-name exp1.yaml ```

We also used a ```bash sweep.yaml``` file with Weights & Biases (WandB) to define Bayesian hyperparameter search. This allowed us to automatically run multiple configurations with different learning rates, optimizers, and batch sizes.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured reproducibility in multiple ways. First, we used configuration files managed by Hydra, so all hyperparameters and settings were stored in YAML format. This allows any experiment to be rerun by specifying the exact same config file, e.g., via ```bash --config-name exp1.yaml```.

To guarantee that no information is lost, all experiments were also logged to Weights & Biases (WandB). Each run logs the hyperparameters, training loss, and validation accuracy across epochs. The logs are automatically stored in the cloud and can be revisited later for comparison or reproduction.

Furthermore, we containerized both the training and preprocessing steps using Docker (```bash train.dockerfile``` and ```bash data.dockerfile```). This ensures that dependencies and environments are consistent across machines and time.

Finally, each trained model was saved locally with a timestamp and uploaded to a Google Cloud Storage bucket. This versioned storage allows us to retrieve the exact model weights used in any experiment, reinforcing reproducibility across both code and data artifacts.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We created three separate Dockerfiles to support the key stages of our pipeline: data processing, model training, and deployment. The ```bash data.dockerfile``` is used to preprocess the Titanic dataset before training with our data script (```bash data.py```), while the ```bash train.dockerfile``` runs our training script (```bash train.py```) and logs the results to Weights & Biases. The ```bash api.dockerfile``` is responsible for running our trained model as a containerized inference API using the Functions Framework, which mimics how we would deploy it on Google Cloud Functions. All Dockerfiles use the same slim base image (```bash ghcr.io/astral-sh/uv:python3.13-bookworm-slim```) and install dependencies using ```bash uv``` and a ```bash pyproject.toml``` with a locked ```bash uv.lock file```, ensuring reproducibility.

To run our API container locally for testing, we used ```bash docker build -f dockerfiles/api.dockerfile -t api .``` followed by ```bash docker run -p 8080:8080 api```, which exposes the API on port 8080.

Link to our docker files: [INDSÆT LINKS HER]

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---