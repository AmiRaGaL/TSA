# Twitter Sentiment Analysis

### Code Files
- **TSA. ipynb**: Contains exploratory data analysis (EDA) and preprocessing steps along with basic model implementations.
- **TSA_ML.ipynb**: Encompasses the entire implementation process.
- **Inference.py**: Includes functions for inference. It can be imported into other files for making predictions.
- **InferenceSL.py**: Consists of Streamlit code for inference. Ensure Streamlit is installed (`pip install streamlit`) and run the file using the command `streamlit run InferenceSL.py` on the command prompt. This will display a user interface where tweets can be passed for making predictions.
- **TSA_LLM.ipynb**: Contains code to load and test a trending Language Model (LLM) in this domain, along with some analysis.

### Folders
- **Results**: Contains all the screenshots of our implementation results.
- **models**: Contains all saved models, tokenizers and vectorizers.

### Dataset Source
[Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

The dataset provided is from Kaggle and is sourced from the "Sentiment140" dataset by Kazanova. This dataset is widely used for sentiment analysis tasks and comprises 1.6 million tweets labeled with sentiments (positive, negative, neutral). It serves as a valuable resource for training and evaluating sentiment analysis models.