# Healthcare-utilization-and-Machine-Learning
These are the codes that I have used to get the results in the paper "Healthcare utilization and its evolution during the  years: building a predictive and interpretable model", the third chapter of my doctoral thesis. The scripts are written in Python.

I am the only author of this paper.

The goal of the paper is to predict and understand the determinants of
healthcare utilization, measured as Number of doctor visits in the last
three months. To do so, I make use of both Supervised and Unsurpevised Machine Learning algorithms.

In particular, I start constructing two data specifications, called the "Pooled" and "Transformed Pooled". In the former, 
I simply pool together observations from multiple years, hence possibly repeating the same individual multiple times.
In the latter, instead, I consider a Mundlak-like transformation to take into account possible time effects.

The predictive accuracies of Random Forests are compared to that of higher-bias algorithms like Linear Regression,
Poisson Regression and Negative Binomial Regression. 

Moreover, to build an intepretable model, I compare the Shapley Values from the Random Forests (and their associated mean absolute values)
with the absolute values of the Coefficients from the Linear Regressions.

Finally, all the above operations are also done in clusters of individuals obtained using K-Means clustering. 
I considered 5 clusters from the Pooled specification and 3 from the Transformed Pooled.

The considered data are from the German Socio-Economic Panel. The codes are havily commented to favour readability.
The paper is in advanced state and on the point to be submitted to journals, hence I am not publishing it here.
