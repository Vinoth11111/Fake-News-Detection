# python image
FROM python:3.10-slim

#set working directory
WORKDIR /app

#COPY requirements file
COPY requirements.txt .

#install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# coping all the project files into the container.
COPY . .

#expose the port(the port streamlit runs on)
EXPOSE 8501

# the command to run the streamlit app
CMD ["streamlit", "run","Fake_news_UI.py", "--server.port=8501", "--server.address=0.0.0.0"]