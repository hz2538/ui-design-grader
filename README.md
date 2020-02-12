# ui-design-grader
AppDecoder V1.0. Insight Data Engineering Project NY 20A

### Getting Started
These instructions will provide you a brief description of my project, and a guideline for our codes as well on how to run those modules on corresponding AWS EC2 instances for development and testing purposes.

### Description 
The objective is to build a data-driven APP design recommendation platform (see the [website](http://www.dataengineer.site/) or the [video demo](https://youtu.be/SUgCPqrCQWE)). The user are required to input the current layout design image accompanied with the labeled semenatic annotation image, and the output would be the recommendation on how to revise your current layout, as well as the refering UI designs in my current database.

### Dataset
The dataset is a large and publicly accessible APP design dataset called RICO (site: http://interactionmining.org/rico). You can download full dataset from that website. The dataset contains several elements: 1. UI Screenshots and View Hierarchies; 2. UI Metadata; 3. UI Layout Vectors; 4. Interaction Traces; 5. Animations; 6. Play Store Metadata; 7. UI Screenshots and Hierarchies with Semantic Annotations. 

(Note: currently, 1, 2, 6, 7 are used in AppDecoder v1.0. The rest would be used in future developments.)

### Pipeline

![pipeline](./documents/pipeline.png)

***Offline Job***: Currently, metadata and view hirearchy files are ingested from S3 bucket into Spark. Spark processes the metadata and saves the static tables into the PostgreSQL database. Also, all UIs' hirearchy files are parsed in Spark into rows of elements information. Those are stacked into one table in PostgresSQL database. Meanwhile, semantic annotation images are participated in model training by Tensorflow 2.0.

***Online Job***: The flask front-end is loaded on the pre-trained model. The user inputs would directly led to the model, and give the generated results and similar candidates. A filter is processed on the candidates to check if the revised elements are contained in the candidates' layout. The qualified candidates are displayed on the platform.

### Introducing the files in project

    |-- flask
        |-- app
            |-- fonts
            |-- static
            |-- templates
            |-- tf2
                |-- tf2vae
                |-- generate.py
                |-- models.py
            |-- __init__.py
            |-- interface.py
            |-- query.py
        |-- testimage
        |-- config.ini
        |-- install.sh
        |-- wsgi.py
        |-- requirements.txt
    |-- spark
        |-- func
            |-- metadata.py
            |-- model.py
            |-- ui_semantic.py
            |-- utils.py
        |-- localdata
        |-- config.ini
        |-- install.sh
        |-- run.py
        |-- save_to_db.py
        |-- requirements.txt
    |-- documents
    |-- LICENSE
    |-- README.md

        
            
*flask frontend*

* flask/app: The folder contains all basic elements such as fonts, static (.css, .js) files, and website templates (.html) and the functions for the server.
    * interface.py
    >contains view functions that mapped to request URLs.
    * query.py
    >contains the connection and query function to PostgreSQL database.  

* flask/app/tf2: The tensorflow 2.0 implementation of the deep learning models. Currently, I include Convolutional AutoEncoder ([article](http://users.cecs.anu.edu.au/~Tom.Gedeon/conf/ABCs2018/paper/ABCs2018_paper_58.pdf)), and Variational AutoEncoder ([article](https://arxiv.org/abs/1312.6114)) models.
    * models.py
    >contains class 'Model' to run tests on current models with customized choice. Mainly AE, VAE, VAE-GAN (not updated yet) models can be chosen.
    * generate.py
    >contains the computer-vision methods that refine the output of generative models. 

* flask/testimage
>contains the test images for users. You can upload those to the website.

* flask/install.sh
>contains the steps of enviromental settings.

* wsgi.py
>the script to import the app package and start the server.

*spark backend*

* spark/func
>contains all functions and classes to load in different format of data.

* install.sh
>contains the steps of enviromental settings.

* run.py
>contains the first implementation of similarity calculation based on SparkML.

* save_to_db.py
>contains the static table storage to PostgreSQL.


### Prerequisites
There are separate computing units utilized to realize the current pipeline. To reproduce my environment, you will need:
* 1 Spark Cluster (4 m5.large nodes) - backend
* 1 Flask Cluster loaded with Tensorflow 2.0 (1~2 p2.xlarge nodes with 1 K80 GPU for each) - frontend
* 1 PostgreSQL Node (1 t2.micro node) - database

For the Spark Cluster, you are required to first install Pegasus. Please see the [installation tutorial](https://github.com/InsightDataScience/pegasus).

[AWSCLI](https://aws.amazon.com/cli/?nc1=h_ls) is also needed to be installed and configured for the cluster. 

Folder 'flask' and 'spark' were run in different envioronments. See full details of the environment requirements in *requirements.txt* inside 'flask' and 'spark' folders.



### How to run the code 

* Make sure you have the spark cluster and the flask cluster ready. Clone this repo to both of them using `git clone https://github.com/hz2538/ui-design-grader.git`.

* For Spark environment, run `spark/install.sh` to install miniconda and other necessary python packages.

* [PostgreSQL setup](https://blog.insightdatascience.com/simply-install-postgresql-58c1e4ebf252) on the t2.micro node.

* Go to the spark repository, aftering editing `config.ini`, run `python save_to_db.py` to save to static tables; run `python run.py` to try the sparkML method.

* For Flask and tensorflow2.0 environment, please launch the instance choosing "Deep Learning AMI (Ubuntu 16.04) Version 26.0". 
    * Run `source activate tensorflow2_p36` to go to the virtual environment. 
    * Run `flask/install.sh` to install additional packages.
    * Run `python wsgi.py` to launch the server.

### Acknowledgements

