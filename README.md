# EECS731_Project02
Classy Shakespeare plays and players

1. Set up a data science project structure in a new git repository in your GitHub account
2. Download the Shakespeare plays dataset from https://www.kaggle.com/kingburrito666/shakespeare-plays
3. Load the data set into panda data frames
4. Formulate one or two ideas on how feature engineering would help the data set to establish additional value using exploratory data analysis
5. Build one or more classification models to determine the player using the other columns as features
6. Document your process and results
7. Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

# Data analysis by using Feature engineering
We can analyze which part of the play it comes from by counting the length of the playerline.
Normally, we know that the middle part of a playerline has longer and more stable char than the last line. （'stable' here means
Several consecutive lines have similar length.）
Therefore, we can jointly predict which player it belongs to by using the length feature of the players line and the players linenumber feature.

# Classification

I pick Decision Tree as the Classifier.

Split 20% for testing set, 80% for training set. Dataset will be shuffle.
feature: Play =============-> predict: Player
acc: 0.22043649850221103

feature: PlayerLinenumber =-> predict: Player
acc: 0.0579145071561029

feature: ActSceneLine =====-> predict: Player
acc: 0.04222338452760211

feature: PlayerLine =======-> predict: Player
acc: 0.02215776710570111
# Conclusion

I use the controlled variable method to compare the results of different column predictions of the player, and found that the prediction using Play is the best, and the prediction using Playerline is the worst.
The main reason is that Playerline has a higher complexity in our process of quantifying features. For this, if we can't quantify it well, it will be difficult to use.
