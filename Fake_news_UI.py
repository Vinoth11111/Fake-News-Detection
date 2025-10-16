import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer


#setting header for the page.
st.header("Fake News Detection")
st.image('/Users/vinothkumar/Downloads/fake.png')

st.subheader('Enter your news article in the text box below to check if it is real or fake.')
#creating a text area for user input.
#user_input = st.text_area("Enter your news article here", height=200)
senetence_model = SentenceTransformer('all-MiniLM-L6-v2')
@st.cache_resource #storing the model and embedding matrix in cache memory to avoid reloading every time.

# creating a function to load the model and embedding matrix.
def load_model():
    with open('/Users/vinothkumar/Documents/Fake_News_Detection/model/fake_news_trained_model','rb') as m:
        model = joblib.load(m)
    return model
trained_model = load_model()
user_input  = st.chat_input("Enter your news article here")
if user_input:
    st.write("Your Article:")
    if user_input.strip() == "":
        st.write("Please enter a valid news article.")
    elif len(user_input) < 100:
        st.write(user_input)
        st.write('article don\'t have enough information or check the lenght of the lenght input article')
    else:
        x = senetence_model.encode([user_input])
        prediction = trained_model.predict(x)
        st.write(user_input)
    
    # Here you would typically call your fake news detection model
    # For demonstration, we'll just display a placeholder result
        st.success('The article is Fake' if prediction[0] ==0 else 'The article is Real')
    