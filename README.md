# Inferring gender of reddit users
Master thesis project. Predicting gender of reddit users based on their comments. 

**BigQuery_queries**  folder contains examples of queries that were used to extract demographic information about Reddit users.

**data** folder contains the list of words with corresponding logistic regression weights and the list of users with demographic labels.

**text2gender** contains the pre-trained models that produce probability between 0 and 1, where 1 is a female user and 0 is male user.

## Example of usage:
```python
from text2gender import Text2Gender
t2g = Text2Gender()  # Initialization
comments = ['My wife and I are going to visit Canada next week', 'Good luck with your exams']
print(t2g.predict_charcnn(comments))
#  Output: [[0.09709139], [0.6048636]]
```


