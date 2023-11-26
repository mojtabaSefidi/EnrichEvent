# EnrichEvent
Official Implementation of "[EnrichEvent: Enriching Social Data with Contextual Information for Emerging Event Extraction](https://arxiv.org/abs/2307.16082)"

### How to run?
1. Go to main.ipynb
2. initialize and customize the parameters based on your usage
3. Run all the cells of main.ipynb
4. The results will be saved in the directory where you have addressed

### How to find the dataset?
1. You can find details of our proposed datasets in /Dataset folder
* Note: You can also use your own dataset but you should adjust its structure and columns' name

### How to find the dataset?
1. You can find details of our proposed datasets in /Dataset folder
* Note: You can also use your own dataset but you should adjust its structure and columns' name

### How to train your own trend detection model?
1. Go to /Trend_Detection folder
2. Use train.py to build and train your own trend detection model
* Note: You need a labeled dataset. You can also use dataset_labeling.py to your dataset based on your proposed key phrases

### How to train your own event summarization model?
1. Go to /Event_Summarization folder
2. Use train.py to build and train your own event summarization model
* Note: You need a pre-trained embedding model based on the language of your dataset

### Note
Feel free to contact me for possible issues and problems
