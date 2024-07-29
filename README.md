# P5-Machine-Learning-API
![The image should showcase the integration of high ](https://github.com/user-attachments/assets/81cdea3a-56a8-4c44-9c1b-2878accb9506)


## Project Overview 📖

In this project, we utilize supervised machine learning techniques, particularly classification, to develop a model that predicts the likelihood of sepsis in patients admitted to Intensive Care Units (ICUs). Our classification model is designed to distinguish between patients at risk of developing sepsis and those who are not, using a comprehensive set of features.

**Sepsis** Sepsis is a life-threatening medical emergency caused by your body’s overwhelming response to an infection. Without urgent treatment, it can lead to tissue damage, organ failure and death.

**Who does sepsis affect?**
Sepsis can affect anyone, but people with any kind of infection, especially bacteremia, are at a particularly high risk.

Other people who are at a high risk include:

- People older than 65 years old, newborns and infants, and pregnant people.
- People with medical conditions such as diabetes, obesity, cancer and kidney disease.
- People with weakened immune systems.
- People who are in the hospital for other medical reasons.
- People with severe injuries, such as large burns or wounds.
- People with catheters, IVs or breathing tubes.

More than 1.7 million people in the United States receive a diagnosis of sepsis each year. There are differences in sepsis rates among different demographic groups. Sepsis is more common among older adults, with incidence increasing with each year after the age of 65 years old.

[Source](https://my.clevelandclinic.org/health/diseases/12361-sepsis)


The project is guided by the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework.

## Project Links :link:

| Notebook      | Docker Image        | Published Article |
|-----------|-------------|:-------------:|
| [Sepsis ML model notebook](https://github.com/bamzyyyy/P5-Building-Machine-Learning-APIs-With-FastAPI/blob/bamzzz/dev/API_App.ipynb) | [Docker image on Docker Hub](https://hub.docker.com/repository/docker/bambam0007/sepsis-app-fast-apii/general) |  [Sepsis Prediction Article](https://medium.com/@obandoandrew8/building-a-machine-learning-api-with-python-fastapi-and-docker-7281df112565)) |


## Table of Contents 🔖
- [Project Overview](#project-overview-)
- [Project Links](#project-links-link)
- [Some Tools Used For The Project](#some-tools-used-for-the-project-hammer_and_wrench)
- [Dataset](#data-fields-)
- [Repository Setup](#repository-setup)
- [Run FastAPI](#run-fastapi)
- [API Screenshots](#fastapi-screenshots)
- [Author](#author-writing_hand)

##  Some Tools Used For The Project :hammer_and_wrench:
<p align="left">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" alt="vscode" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original-wordmark.svg" alt="pandas" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" alt="numpy" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="python" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" alt="jupyter" width="45" height="45"/>
<img src="https://icongr.am/devicon/docker-original-wordmark.svg?size=45&color=currentColor" alt="docker"/>
</p>


## Data Fields 💾

| Column   Name                | Attribute/Target | Description                                                                                                                                                                                                  |
|------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ID                           | N/A              | Unique number to represent patient ID                                                                                                                                                                        |
| PRG           | Attribute1       |  Plasma glucose|
| PL               | Attribute 2     |   Blood Work Result-1 (mu U/ml)                                                                                                                                                |
| PR              | Attribute 3      | Blood Pressure (mm Hg)|
| SK              | Attribute 4      | Blood Work Result-2 (mm)|
| TS             | Attribute 5      |     Blood Work Result-3 (mu U/ml)|                                                                                  
| M11     | Attribute 6    |  Body mass index (weight in kg/(height in m)^2|
| BD2             | Attribute 7     |   Blood Work Result-4 (mu U/ml)|
| Age              | Attribute 8      |    patients age  (years)|
| Insurance | N/A     | If a patient holds a valid insurance card|
| Sepssis                 | Target           | Positive: if a patient in ICU will develop a sepsis , and Negative: otherwise |

## Repository Setup

Install the required packages to be able to run the API locally.

You need to have [`Python 3`](https://www.python.org/) on your system. Then you can clone this repo and being at the repo's `root :: repository_name> ...`  follow the steps below:

- Windows:
        
        python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

- Linux & MacOs:
        
        python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

The two long command-lines have the same structure. They pipe multiple commands using the symbol ` ; ` but you can manually execute them one after the other.

1. **Create the Python's virtual environment** that isolates the required libraries of the project to avoid conflicts;
2. **Activate the Python's virtual environment** so that the Python kernel & libraries will be those of the isolated environment;
3. **Upgrade Pip, the installed libraries/packages manager** to have the up-to-date version that will work correctly;
4. **Install the required libraries/packages** listed in the `requirements.txt` file so that they can be imported into the python script and notebook without any issue.

**NB:** For MacOs users, please install `Xcode` if you have an issue.

## Run FastAPI

- Run the API (being at the repository root):
        
  FastAPI:
    
    - Main

          cd src, uvicorn myApp:app --reload 

    <!-- - Sepsis prediction

          cd src, uvicorn myApp:app --reload  -->


  - Go to your browser at the local port, to explore the API's documentation :
        
      http://127.0.0.1:8000/docs

Here is a [tutorial](https://fastapi.tiangolo.com/tutorial/) for fastAPI

## FastAPI Screenshots

- App documentation

<img width="1419" alt="Screenshot 2024-07-28 at 8 31 35 AM" src="https://github.com/user-attachments/assets/71f4c438-17f9-4045-972d-d8f3dec8d4b2">


- Input
<img width="1394" alt="Screenshot 2024-07-28 at 8 35 23 AM" src="https://github.com/user-attachments/assets/342b21e0-a01e-4d04-b9c4-754079f755a0">


- Prediction
<img width="1384" alt="Screenshot 2024-07-28 at 8 35 37 AM" src="https://github.com/user-attachments/assets/60b57483-0d01-4a4c-b2c7-09502b77fd7a">

## Authors 👨‍💼

  | Name                                            | LinkedIn                                                                                                                                                                                                                                               | Medium Article |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
|Victor Nyarko Anim         |[Profile](https://www.linkedin.com/in/victor-anim-83115818/)|[Article](https://medium.com/@victor.nyarko/my-full-stack-app-using-fastapi-as-my-backend-and-streamlit-as-my-frontend-containerization-with-14ea216914ec)
| Aminu Oluwarotimi Desmond | [Aminu Oluwarotimi Desmond](https://www.linkedin.com/in/aminudesmond/) |[Sepsis Prediction Web App: FastAPI Development and Docker Containerization](https://medium.com/@aminuoluwarotimi/building-a-machine-learning-api-with-python-fastapi-and-docker-81ec608b9263)|


## Model Training and Saving ⏳

The Xgboost, Logistic Regression, SVM and Gradient boost model was trained using the Telecommunication Customer Churn as shown in this GitHub repository: [P5-Building-Machine-Learning-APIs-With-FastAPI
](https://github.com/bamzyyyy/P5-Building-Machine-Learning-APIs-With-FastAPI).

## Acknowledgments 🙏

I would like to express my gratitude to the [Azubi Africa Data Analyst Program](https://www.azubiafrica.org/data-analytics) for their support and for offering valuable projects as part of this program. Not forgeting my scrum masters on this project [Rachel Appiah-Kubi](https://www.linkedin.com/in/racheal-appiah-kubi/) & [Emmanuel Koupoh](https://github.com/eaedk)

## License 📜

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact 📧

For questions, feedback, and collaborations, please contact [Aminu Oluwarotimi Desmond](aminuoluwarotimi@gmail.com).
