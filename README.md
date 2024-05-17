# RECS: Real-time Emotion Recognition using Physiological signals in e-Learning 
- `ABSTRACT:` In face-to-face and online learning, emotions and emotional intelligence have an influence and play an essential role. Learners’ emotions are crucial for e-learning system because they promote or restrain the learning. Many researchers have investigated the impacts of emotions in enhancing and maximizing e-learning outcomes. Several machine learning and deep learning approaches have also been proposed to achieve this goal. All such approaches are suitable for an offline mode, where the data for emotion classification are stored and can be accessed infinitely. However, these offline mode approaches are inappropriate for real-time emotion classification when the data are coming in a continuous stream and data can be seen to the model at once only. We also need real-time responses according to the emotional state. For this, we propose a real-time emotion classification system (RECS)-based Logistic Regression (LR) trained in an online fashion using the Stochastic Gradient Descent (SGD) algorithm. The proposed RECS is capable of classifying emotions in real-time by training the model in an online fashion using an EEG signal stream. To validate the performance of RECS, we have used the DEAP data set, which is the most widely used benchmark data set for emotion classification. The results show that the proposed approach can effectively classify emotions in real-time from the EEG data stream, which achieved a better accuracy and F1-score than other offline and online approaches. The developed real-time emotion classification system is analyzed in an e-learning context scenario.

## DATASET
- `DEAP dataset` is required. 
- The experiment is conducted using the `EEG measurements taken from DEAP dataset`. 
- To download `DEAP dataset` click on : https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html

## DATA Rearrangement required
```diff
- CAUTION

+ The DEAP data needs a simple rearrangement to work with the code. 

@@  Check the `data_rearrangements` folder for the  DEAP  data rearrangement from the .dat or .mat file from the DEAP dataset. @@
@@ Then follow the follwoing steps. @@

```


## Installation 
- Programming language
  - `Python 3.6 or above (tested in 3.10)`

- Operating system (tested)
  - `Ubuntu 18.04 (64 bit) - Intel CPU`
  - `MAC mini M1 - ARM based CPU`

- Required packages
  - `Scikit-Learn and River` &#8592; for model's performance matrics.
  - `Numpy` &#8592; for RECS's model development.
  - `River` &#8592; for streaming model development.
  - `Scikit-Learn` &#8592; for offline ML model development.
  
- Installation steps:
  - Step 1: Install `Anaconda`. 
  - Step 2: Create a `virtual environment` in Anaconnda 
  - Install the required packages using `pip` from the `requirements.txt` file.
  - Step 3: Open `terminal`, and `activate environment`.
  - Step 4: Run files :wink:.


## Publication

- The developed system is called **"Real-time Emotion Classification System (RECS)"**. The RECS is developed using Logistic Regression (LR) optimized with Stochastic Gradient Descent (SGD) in online mode.

- This work is published in **MDPI Sensors Journal**. The link to the paper "**Real-Time Emotion Classification Using EEG Data Stream in E-Learning Contexts**" is : https://doi.org/10.3390/s21051589. 
  
  
 **Please cite the paper using the following bibtex:**
  
    @Article{s21051589,
      AUTHOR = {Nandi, Arijit and Xhafa, Fatos and Subirats, Laia and Fort, Santi},
      TITLE = {Real-Time Emotion Classification Using EEG Data Stream in E-Learning Contexts},
      JOURNAL = {Sensors},
      VOLUME = {21},
      YEAR = {2021},
      NUMBER = {5},
      ARTICLE-NUMBER = {1589},
      URL = {https://www.mdpi.com/1424-8220/21/5/1589},
      ISSN = {1424-8220},
      DOI = {10.3390/s21051589}
    }

  **MDPI and ACS Style**
  ```
    Nandi, A.; Xhafa, F.; Subirats, L.; Fort, S. Real-Time Emotion Classification Using EEG Data Stream in E-Learning Contexts. Sensors 2021, 21, 1589. https://doi.org/10.3390/s21051589
```
  **AMA Style**
```  
    Nandi A, Xhafa F, Subirats L, Fort S. Real-Time Emotion Classification Using EEG Data Stream in E-Learning Contexts. Sensors. 2021; 21(5):1589. https://doi.org/10.3390/s21051589
```   
    
    
# NOTE*: Please feel free to use the code by giving proper citation and star to this repository.

## 📝 License

Copyright © [Arijit](https://github.com/officialarijit).
This project is MIT licensed.
