Team members: Erica, Nicole, Maria

Our Dashboard https://homerunscientist.streamlit.app/


## Prerequisities

Before you begin, ensure you have met the following requirements:

* You have a _Windows/Linux/Mac_ machine running [Python 3.6+](https://www.python.org/).
* You have installed the latest versions of [`pip`](https://pip.pypa.io/en/stable/installing/) and [`virtualenv`](https://virtualenv.pypa.io/en/stable/installation/) or `conda` ([Anaconda](https://www.anaconda.com/distribution/)).


## Setup

To install the dependencies, you can simply follow this steps.

Clone the project repository:
```bash
git clone https://github.com/MariaOros/Hackathon2024-baseball.git
cd Hackathon2024-baseball
```

To create and activate the virtual environment, follow these steps:

**Using `conda`**

```bash
$ conda create -n streamlit python=3.7

# Activate the virtual environment:
$ conda activate streamlit

# To deactivate (when you're done):
(streamlit)$ conda deactivate
```

**Using `virtualenv`**

```bash
# In this case I'm supposing that your latest python3 version is 3.7
$ virtualenv streamlit --python=python3

# Activate the virtual environment:
$ source streamlit/bin/activate

# To deactivate (when you're done):
(streamlit)$ deactivate
```

To install the requirements using `pip`, once the virtual environment is active:
```bash
(streamlit)$ pip install -r requirements.txt
```

#### Running the script

Finally, if you want to run the main script:
```bash
(streamlit)$ streamlit run app.py
```
