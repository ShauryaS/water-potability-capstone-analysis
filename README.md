### Water Potability Analysis

**Author**
Shaurya Srivastava

#### Executive summary

#### Rationale
This question holds importance due to the essential nature of water for human survival. Despite its abundance, not all water sources are suitable for consumption. Distinguishing potable water from non-potable sources is challenging. Developing a high-precision model classifier that identifies key indicators of water potability would greatly assist individuals in discerning when it is safe to drink from water sources.

#### Research Question
With limited access to water in certain regions, can we accurately and precisely predict if a water source is drinkable based on water properties? Which properties contribute the most to whether water is drinkable or not?

#### Data Sources
The data source is a dataset containing water portability data from Kaggle. [Link to dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability).

#### Methodology
This is a classification problem that will determine whether water with certain properties is potable (1) or not potable (0). Some models I plan to explore are Logistic Regression, Decision Trees, KNN, SVMs, and Random Forest. I expect to follow CRISP-DM technique, exploratory data analysis, and data pipelines. 

#### Results
After conduction exploratory data analysis and building classifier models, it can be conclused that we cannot accurately or precisely predict if a water source is drinkable based on water properties. I trained 5 models: Logistic Regression, KNN, Decision Tree, SVM, and Random Forest. All 5 models had an accuracy under 70%. The best model is the Random Forest classifier with an accuracy of ~65% and precision of ~71%. However, this model's accuracy and precision are not good enough as people's lives could be at risk when drinking bad water due to an incorrect classification.

#### Next steps
Based on the results of the exploratory data analysis and the performance of the classifier models, here are some suggestions for potential next steps:
- Collect more accurate data of water sources and whether it is potable. 
    - Good data and numerous data seems to be the limiting factor for highly performant classifier models that predict with high accuracy and precision if water is potable
- Use modern methods of machine learning (i.e. neural networks) to see if classification is better
- Try multiple random states for the train test split and the classifier models

#### Outline of project

- [Link to notebook](./water_potability_analysis.ipynb)

#### Contact and Further Information

Feel free to contact me at my [LinkedIn](https://www.linkedin.com/in/shauryas481/) if you have questions or comments. 