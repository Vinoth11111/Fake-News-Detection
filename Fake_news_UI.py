import streamlit as st


#setting header for the page.
st.header("Fake News Detection")
st.subheader('Enter your news article in the text box below to check if it is real or fake.')
#creating a text area for user input.
#user_input = st.text_area("Enter your news article here", height=200)
user_input  = st.chat_input("Enter your news article here")
if user_input:
    st.write("You entered:")
    st.write(user_input)
    # Here you would typically call your fake news detection model
    # For demonstration, we'll just display a placeholder result
    st.success("This is a placeholder result. Replace with actual model output.")