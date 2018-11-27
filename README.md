# Enhancing Starbucks Customer Experience by Building Recommendation Engines
This is a data science project that explore Starbucks App simulated customer purchase behavior data. The goal of this project is to create a promotion offer recommendation engine for the users.

There are also a four part article published on Medium. You can find the articles here:  
[Part 1](https://medium.com/@agustinus.thehub/enhancing-starbucks-customer-experience-by-building-recommendation-engines-part-1-108ddd1d729)
[Part 2](https://medium.com/@agustinus.thehub/enhancing-starbucks-customer-experience-by-building-recommendation-engines-part-2-7703cf332767)
[Part 3](https://medium.com/@agustinus.thehub/enhancing-starbucks-customer-experience-by-building-recommendation-engines-part-3-45100f085daf)
[Part 4](https://medium.com/@agustinus.thehub/enhancing-starbucks-customer-experience-by-building-recommendation-engines-part-4-f67b9a45fc19)

# Installation
Clone this repo: https://github.com/nugroho1234/starbucks-project.git
I used python 3.6 to create this project and the libraries I used are:
1. Pandas
2. Numpy
3. Matplotlib
4. Scikit-learn
5. Math
6. Json
7. Time

# Project Motivation
Apart from the completion of the project, my main motivation to create this project is based on my love for Starbucks coffee. When I was given the chance to create something out of the data, I didn't think twice. I decided to create two recommendation engines for Starbucks App. One is based on segment clusters, and the other one is based on user similarities.

The tasks involved in creating the recommendation engines are:
1. Clean the data sets given since it contains a lot of NaN values and a couple of mistakes.
2. Explore and visualize the data to get a good picture of the demographic profile as well as the purchase behavior of the customers.
3. Engineer features which were used in creating the recommendation engines. The features generated were ratio data of the offer-purchase behavior.
4. Create customer segmentation based on the features. This was done using KMeans clustering.
5. Create recommendation engine based on clusters. This takes a user id, map the user id to the cluster this user belongs to, and recommend top 3 offers which the cluster members like.
6. create recommendation engine based on user similarities. This takes a user id, search for 10 users similar to the id, and recommend top 3 offers these 10 users like best.

# File Descriptions
### Starbucks_Capstone_notebook.ipynb
This is the file that describes the steps I took in creating the recommendation engines. The code I used to solve the tasks I mentioned in the Project Motivation section is in this file.
### The .csv files
These are the files that was generated during the data cleaning phase.
1. master_df.csv - This is the initial combination of 3 data sets mentioned above.
2. df_use.csv - This is the data that contains initial variables and dummy variables
### The .json files
These files are located in the 'data' folder. The folder contains 3 files:
1. portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
2. profile.json - demographic data for each customer
3. transcript.json - records for transactions, offers received, offers viewed, and offers completed

# How to Use This Project
This project is aimed to create a recommendation engine. You can use the functions I created to create a .py file, and create another .py file with the recommendation results. The recommendation results are still in the form of list. If you wish to deploy it, you can change it to json so that it can be used in another app using API.

## Data Cleaning
During the data cleaning phase, I conducted several tasks:
1. Divide channels column in portfolio data frame into 4 dummy columns: web, email, mobile, and social.
2. Create a dictionary with offer id as keys and aliases for the purpose of simplifying the merging of 3 data sets later.
3. Merging profile and transactions data frames.
4. Map the value column in the merged data frame into offer_id, transaction_amount, transaction, and reward_achieved.
5. Check whether transactions were caused by offers. If it were, I created a column called transaction_from_offer and assign 1, and another column called transaction_offer_viewed and assign the offer id. If it weren't, assign 0 to transaction_from_offer, and assign 'no transaction' to transaction_offer_viewed.
6. Predicting age and income which contains wrong values and NaN values by using Random Forest Regressor.
7. Classifying gender which contains None values by using Random Forest Classifier

## Feature Engineering
To engineer the features that are going to be used in the recommendation engine, I will do the following things:
1. Replace some of the 'no transaction' values in the transaction_offer_viewed column of df_use dataframe to transaction_not_from_offer. The values replaced are the ones with the value of 1 in transaction column of df_use dataframe.
2. Create dummy variables out of transaction_offer_viewed column
3. Create a new dataframe, user_demographic, which contains the demographic variables and the transaction behavior of the users.
4. Create 3 new columns which focus on ratio, percentage of offer completed, percentage of transaction after offer received, and percentage of transaction happens although user received offers.
5. Create a new data frame called user_demographics which stores the user demographic variables as well as offer-purchase behavior.

## Data Exploration / Visualization
The code I used to visualize the data can be found in the notebook. The results of the exploration are:
1. Most of the app users come from lower-middle class although Starbucks items are not cheap.
2. Users who joined on earlier years might have discontinued their memberships.
3. Users who joined on the year 2016 made the most average amount of transaction
4. Users who joined on the year 2017 made the most amount of transaction, although individually they don’t really spend a lot.
5. Female users tend to spend more to get the rewards offered.
6. Most of the app users are male, but they don’t really try to complete an offer should they view it.

## Clustering
I used standard scaler to scale the data. This is necessary because there are columns with really big values and those with just 0 and 1. In other words, there are big differences in variance for each feature that were used in clustering.

The method of choosing how many clusters (k) I used was the elbow method. After computing and plotting the SSE vs K, I found that I should use 16 clusters to segment the Starbucks App users.

The description of the first 3 clusters are as follows:
1. The first cluster is populated with Male customers with the age around 49 and the income around USD 58052. They tend to favor bogo4 over other types of offer. After receiving an offer, they do about 2-3 transactions. They also do moderate transaction not from offer received, indicating that they do like to have their coffee at Starbucks.
2. The second cluster is also populated with Male customers with the age around 49 and the income around USD 56291. They tend to favor discount3 over other types of offer. After receiving an offer, they only do about 1 transaction. However, they do purchase in Starbucks although they didn't receive any offer. I think, these customers are not really loyal. They buy coffee whenever they feel like it, and they prefer discount over anything else.
3. The third cluster is populated with Male customers with the age around 60 and income around USD 78141. They tend to favor bogo1 over other types of offer. After receiving an offer, they do about 2-3 transactions and they also buy coffee although they didn't receive offers. This cluster is similar to the first cluster, only they have higher income. 

## Recommendation based on Clusters
The tasks I did are as follows:
1. I have to consider which users don't like offers. Therefore, I will create a dataframe containing the user_id and transaction not from offer ratio. If the ratio is greater than the mean value, I will not recommend anything to these users.
2. Create cluster-promotion matrix.
3. Check whether a user likes to be given promotion or not. If they don't like promotion, do not recommend anything.
3. If they like promotion, check which cluster a user belongs to.
4. Recommend top 3 offers which the cluster this user belongs to like, and get the list of these offer ids.  
5. If the user id is new, I will give a list of most popular offers.

## Recommendation based on user similarities
The tasks I did are as follows:
1. Take an input of user id
2. If the user id does not like promotions, do not recommend anything.
3. If the user id likes promotiion, find 10 most similar users to this user id.
4. Using the user-promotion matrix, recommend top 3 offers which the 10 most similar users like the most.
5. Convert the offer alias into the original offer id.


## Measuring the Effectiveness of the Recommendation Engine
Although it's not possible to directly measure the effectiveness of the engine, Starbucks can always do an A/B testing for a certain period of time. Starbucks has to assign cookies in the app to divide users into 3 groups, 1 control and 2 experiment groups. The control group will not be given this recommendation engine, and the first experiment group will receive the cluster-based recommendation engine, and the second experiment group will receive the user-based recommendation engine. The metrics to watch are:

**Invariant Metrics**
1. The amount of user in control and experiment group. This should be at least similar, close to 50:50.
2. The number of cookies assigned to each group.

**Evaluation Metrics**
1. The average transaction amount. If the experiment group has higher average transaction amount, the recommendation engine can be considered useful.
2. Percentage of offer completed. If the experiment group has higher percentage of offer completed, the recommendation engine can be considered useful.
3. Ratio of transaction from offer and offer received. The higher it is for the experiment group, the better.
4. Ratio of transaction not from offer and offer received. The lower it is for the experiment group, the better.

# Licensing, Author, Acknowledgements
I made this project to create recommendation engines for Starbucks Apps. If you can improve this, I will be glad to hear from you! Feel free to use / tweak this and mention me so I can take a look at your work.

As usual, I benefit greatly from Udacity, stackoverflow and sklearn documentations. I won't be able to live without them :)

