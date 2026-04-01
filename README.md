# 🎬 Netflix Movies and Tv Shows - ML Exploratory-Data-Analysis-Clustering
 
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg" width="200">
</p>

## 📌 Project Overview

Netflix, one of the world’s leading OTT streaming platforms, hosts a vast and diverse collection of **Movies** and **TV Shows** across various countries, genres, and languages.
The goal of this project is to perform **Exploratory Data Analysis (EDA)** and apply **Machine Learning (K-Means Clustering)** to uncover hidden patterns and similarities among Netflix titles.

---

## 🎯 Business Objective

To analyze Netflix’s dataset and identify:

* Trends in content distribution (by **country**, **type**, and **genre**)
* Insights into **ratings**, **release years**, and **durations**
* Clusters of similar content using **unsupervised learning (K-Means)**

This helps Netflix understand:

* Regional and genre-based preferences
* Production and licensing opportunities
* Patterns useful for **recommendation systems**

---

## 🧠 Machine Learning Objective

Implement an **unsupervised learning model** using **K-Means Clustering** to group content based on textual and categorical features such as description, genre, and type.

---
## Problem Statement
  This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset is collected from Flixable which is a third-party Netflix search engine.
     In 2018, they released an interesting report which shows that the number of TV shows on Netflix has nearly tripled since 2010.
     The streaming service’s number of movies has decreased by more than 2,000 titles since 2010, while its number of TV shows has nearly tripled.
     It will be interesting to explore what all other insights can be obtained from the same dataset.
     Integrating this dataset with other external datasets such as IMDB ratings, rotten tomatoes can also provide many interesting findings.
     In this project, you are required to do :
     - Exploratory Data Analysis
     - Understanding what type content is available in different countries
     - Is Netflix has increasingly focusing on TV rather than movies in recent years.
     - Clustering similar content by matching text-based features

## 🛠️ Tools & Technologies

| Category                 | Tools / Libraries              |
| ------------------------ | ------------------------------ |
| Language                 | Python                         |
| Data Handling            | Pandas, NumPy                  |
| Visualization            | Matplotlib, Seaborn, WordCloud |
| Machine Learning         | Scikit-learn                   |
| Text Vectorization       | TF-IDF Vectorizer              |
| Dimensionality Reduction | PCA                            |

---

## 🧾 Key Learnings

*   **Performed EDA to extract meaningful insights** about Netflix's content trends, geographic leads, and the 2020 strategic shift toward TV shows.
*   **Built an unsupervised clustering model** using K-Means and Hierarchical Clustering to group over 7,000 titles into 5 distinct thematic categories.
*   **Understood text vectorization and dimensionality reduction** by implementing TF-IDF for description analysis and PCA to handle high-dimensional matrices while retaining 95% variance.
*   **Learned practical end-to-end data science workflow** encompassing data cleaning, IQR-based outlier treatment, feature engineering, and model deployment.
*   **Validated clustering consistency** by training supervised models like Logistic Regression and XGBoost, achieving up to 100% accuracy in cluster prediction.
*   **Mastered NLP preprocessing techniques** including lemmatization and the removal of stopwords, punctuation, and URLs from raw text descriptions.
*   **Evaluated model performance** using advanced metrics such as the Silhouette Score and Calinski-Harabasz score to ensure optimal cluster separation.

---
     
## 📊 Dataset Description

Attribute Information:
- **show_id**: Unique ID for every Movie / Tv Show
- **type**: Identifier - A Movie or TV Show
- **title**: Title of the content
- **director**: Director of the content
- **cast**: Actors involved
- **country**: Country of production
- **date_added**: Date it was added on Netflix
- **release_year**: Actual Release year
- **rating**: TV Rating
- **duration**: Total Duration (minutes or seasons)
- **listed_in**: Genres (Renamed to 'geners')
- **description**: Summary description

---
## ⚙️ The ML Pipeline
1.  **Data Cleaning:** Handled missing values (substituted with "Unavailable") and treated `release_year` outliers using **IQR**.
2.  **Feature Engineering:** Created new features like `year_added`, `month_added`, `Title_Age`, and `Lexical_Richness`.
3.  **Textual Preprocessing:** Removed punctuation, stopwords, whitespace, and URLs; followed by **Lemmatization**.
4.  **Vectorization:** Applied **TF-IDF** to extract semantic information from content descriptions.
5.  **Dimensionality Reduction:** Used **PCA** to reduce high-dimensional text data while retaining 95% variance.
6.  **Clustering:** Applied **K-Means** with the **Elbow Method** and **Silhouette Score** to identify 5 distinct content clusters.
7.  **Supervised Learning:** Validated clusters using Logistic Regression, Random Forest, and XGBoost.
8.  **Model Tuning:** Used **GridSearchCV (5-fold)** for hyperparameter optimization and cross-validation.

---

## ⚙️ Data Cleaning and Data Preprocessing

[1] **Handling Duplicate Values**: Dataset contained no duplicated values.  
[2] **Handling Null / Missing Values**: 
- Missing values in **director**, **cast**, and **country** were substituted with "Director Unavailable", "Cast Unavailable", and "Country Unavailable".
- Dropped rows with null values in **date_added** and **rating** due to their low count.  
[3] **Handling Outliers**: Outliers in `release_year` were treated using the **Interquartile Range (IQR)** method.  
[4] **Feature Engineering**: 
- Converted `date_added` to datetime and generated `year_added`, `month_added`, and `day_added`.
- Renamed `listed_in` to `geners`.
- Changed `release_year` data type from float64 to int64.  
[5] **Textual Data Preprocessing**: 
- Processed `description` by removing punctuation, stopwords, whitespace, URLs, and special characters.
- Text was vectorized using **TfidfVectorizer** after lemmanization.

---

## 📈 Exploratory Data Analysis (EDA) Insights

The following graphs and plots were primarily created using Matplotlib and the Seaborn package:
- Bar plot, count plot, pair plot, dist plot, box plot, pie plot, and heatmap

### **Key Conclusions from EDA:**

*   **Content Split:** More movies (69.14%) than TV shows (30.86%) are available on Netflix.
*   **Release Windows:** The majority of Netflix movies were released between 2015 and 2020, and the majority of Netflix TV shows were released between 2018 and 2020.
*   **Peak Release Years:** The most movies and TV shows were released for public viewing on Netflix in 2017 and 2020, respectively, out of all released years.
*   **Strategic Pivot:** From 2006 to 2019 Netflix is constantly releasing more new movies than TV shows, but in 2020, it released more TV shows than new movies, indicating that Netflix has been increasingly focusing on TV rather than movies in recent years.
*   **TV Show Expansion:** More TV shows will be released for public viewing in 2020 and 2021 than at any other time in the history of Netflix.
*   **Maturity Ratings:** The majority of TV shows and movies available on Netflix have a TV-MA rating, with a TV-14 rating coming in second.
*   **Recent Additions:** The majority of movies added to Netflix in 2019 and the majority of TV shows added to Netflix in 2020.
*   **2019 Growth:** In 2019, Netflix added nearly one-fourth (27.71%) of all content (TV shows and movies).
*   **Monthly Trends:** The majority of the content added to Netflix was in October and January, respectively, but almost all months throughout the year saw Netflix adding content to its platform.
*   **Production Hubs (Movies):** The majority of movies available on Netflix are produced in the United States, with India coming in second.
*   **Production Hubs (TV Shows):** The United States and the United Kingdom are the two countries that produced the most of the TV shows that are available on Netflix.
*   **Top Movie Directors:** Raul Campos and Jan Suter directed most of the movies available on Netflix for public viewing.
*   **Top TV Directors:** Alastair Fothergill directed most of the TV shows available on Netflix for public viewing.
*   **Popular Genres:** International movies and the second-most popular dramas are available on Netflix as content.
*   **Frequent Cast:** Actors who have appeared in films and TV shows that are most available on Netflix are Lee, Michel, David, Jhon, and James.
*   **Temporal Correlation:** We see that the movie or TV show release year and day of the month on movies or TV shows added to Netflix are slightly correlated with each other.
*   **Content Scaling:** Based on the plot of release_year and year_added, we can conclude that Netflix is increasingly adding and releasing movies and TV shows over time.
*   **Consistency:** We can conclude from plot release_year and month_added that Netflix releases movies and TV shows throughout the all months of the year.

---

## 🤖 Machine Learning Model – K-Means Clustering

**Goal:** Identify content clusters based on textual and metadata similarity.

**Steps:**

1.  **Text Preprocessing:** Cleaned description and genre data (stopwords removal, lowercase conversion, punctuation removal).
2.  **Vectorization:** Transformed processed text into numerical format using **TF-IDF Vectorizer**.
3.  **Dimensionality Reduction:** Applied **PCA (Principal Component Analysis)** to reduce feature space while retaining 95% variance.
4.  **Optimal Clusters:** Determined the ideal number of clusters ($k=5$) using the **Elbow Method** and **Silhouette Score**.
5.  **Clustering:** Applied **K-Means Clustering** to group similar content.

**Results:**
The model successfully identified 5 major clusters representing distinct content categories:
*   **Cluster 0:** International Movies & Dramas
*   **Cluster 1:** Documentaries
*   **Cluster 2:** Stand-Up Comedy
*   **Cluster 3:** Kids' TV & Family Movies
*   **Cluster 4:** Action, Thriller & Sci-Fi

---

## 🛠️ Supervised Learning & Model Evaluation

After clustering, we used the cluster labels as targets for supervised classification to validate the consistency of the groupings.

### **1. Logistic Regression**
*   **Accuracy:** **1.00 (100%)**
*   **Insight:** The perfect score indicates that the clusters are linearly separable in the reduced feature space.

### **2. Random Forest Classifier**
*   **Accuracy:** **0.96 (96%)**
*   **Insight:** Excellent at capturing non-linear relationships. It correctly identified almost all content types, with slight overlap in high-variance clusters.

### **3. XGBoost Classifier**
*   **Accuracy:** **0.99 (99%)**
*   **Insight:** The iterative boosting approach provided near-perfect classification, confirming the mathematical robustness of the K-Means clusters.

---


## 🚀 Future Scope

* Enhance clustering using **Word2Vec** or **BERT embeddings**
* Build a **recommendation system** based on content similarity
* Deploy interactive visualization using **Streamlit / Power BI**

---

## 💡 Conclusion

This project focused on analyzing and clustering **Netflix titles** based on their descriptions, genres, and other attributes using Machine Learning techniques.
A complete end-to-end ML pipeline was developed — starting from **data preprocessing** to **feature engineering, model training, hyperparameter tuning, and deployment**.

The workflow included:

* **Data Cleaning and Preprocessing:** Removal of missing values, text cleaning, and encoding categorical data.
* **Feature Engineering:** Creation of new features such as *Title_Age*, *Lexical_Richness*, and *Num_Genres*.
* **Text Vectorization:** TF-IDF was applied to extract semantic information from descriptions.
* **Dimensionality Reduction (PCA):** Reduced high-dimensional text data while retaining 95% variance.
* **Feature Selection & Scaling:** Eliminated redundant features and standardized all numeric variables.
* **Clustering (K-Means):** Identified meaningful content clusters representing different content types or themes.
* **Model Building:** Developed and compared three models — Logistic Regression, Random Forest, and XGBoost.
* **Model Tuning & Validation:** Used GridSearchCV (5-fold) for cross-validation and hyperparameter optimization.
* **Model Deployment:** Saved the best model in `.joblib` and `.pkl` formats and validated it through a sanity check.

---

### **Final Model Comparison Summary**

| **Model**                 | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **Remarks**                                                       |
| ------------------------- | ------------ | ------------- | ---------- | ------------ | ----------------------------------------------------------------- |
| Logistic Regression       | 1.00         | 1.00          | 1.00       | 1.00         | Perfect baseline; best for linearly separable data                |
| Random Forest             | 0.9908       | 0.98          | 0.98       | 0.98         | Strong model; handles non-linear patterns well                    |
| **XGBoost (Final Model)** | **0.9983**   | **1.00**      | **1.00**   | **1.00**     | **Best performer; highly accurate, robust, and well-generalized** |

---

### **Final Model Selection**

The **XGBoost Classifier** was chosen as the final prediction model because it consistently delivered the best performance across all metrics.
It provided a perfect balance between **accuracy, precision, recall, and generalization**, confirming its reliability for real-world deployment.
The model successfully captured complex relationships in the data and maintained exceptional performance even after cross-validation and hyperparameter tuning.

---

### **Business Impact**

The developed ML model provides actionable insights that directly align with Netflix’s business objectives:

* **Improved Content Clustering:** Helps group similar titles together, supporting better catalog organization.
* **Enhanced Recommendation Systems:** Enables more accurate and personalized content suggestions.
* **Audience Insights:** Helps Netflix understand viewer preferences by analyzing patterns in clustered content.
* **Data-Driven Marketing:** Allows targeted promotions for specific content categories or audience groups.

Overall, the model ensures **higher customer satisfaction, improved engagement, and optimized marketing strategies**, contributing to long-term platform growth and competitive advantage.

---

## 📌 Summary & Conclusions

> The project successfully implemented a complete machine learning pipeline to cluster and predict Netflix titles using advanced algorithms.
> After comparing multiple models, **XGBoost** emerged as the most efficient and accurate, achieving **99.83% accuracy** with perfect precision and recall.
> The model was fine-tuned, validated, saved, and tested for deployment, ensuring it is production-ready.
> This end-to-end ML workflow demonstrates a robust, scalable, and business-relevant solution for **content categorization and recommendation**.
___
