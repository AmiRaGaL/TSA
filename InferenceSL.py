import streamlit as st
import pandas as pd
#from your_model_module import sentiment_analysis_function
from Inference import predict_sentiment, load_models

def get_majority_prediction(predictions):
    # Calculate the majority prediction
    positive_count = predictions.count('Positive')
    negative_count = predictions.count('Negative')
    neutral_count = predictions.count('Neutral')
    
    if positive_count >= negative_count and positive_count >= neutral_count:
        return 'Positive'
    elif negative_count >= positive_count and negative_count >= neutral_count:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    st.title('Sentiment Analysis on Tweets')
    
    # Get user input text
    user_text = st.text_input('Enter your text:')
    
    # Button to trigger sentiment analysis
    if st.button('Analyze Sentiment'):
        # Call your sentiment analysis function to get predictions
        #predictions = sentiment_analysis_function(user_text)
        
        # Display individual model predictions
        st.subheader('Individual Model Predictions:')

        models = ["LR","BNB","SVM"]
        text = "I love the music" # Positive Text
        for model in models:
            if model == "LR":
                vectoriser, lr = load_models(model)
            elif model == "BNB":
                vectoriser, nb = load_models(model)
            elif model == "SVM":
                vectoriser, svc = load_models(model)

        # predictions = predict_sentiment(vectoriser,models,[text])
        # df = pd.DataFrame({'Model': ['Model 1', 'Model 2', 'Model 3'], 'Prediction': predictions})
        df = predict_sentiment(vectoriser,models,[text])
        predictions = df['sentiment'].tolist()
        st.write(df)
        
        # Get majority prediction
        majority_prediction = get_majority_prediction(predictions)
        
        # Display majority prediction
        st.subheader('Majority Prediction:')
        st.write(majority_prediction)

if __name__ == '__main__':
    main()
