FROM python:3.8-slim

WORKDIR /app

# Update this line with the new file name
COPY flight_price_ML_model.py .  
COPY app.py .
COPY requirements.txt .

# Add all the model files
COPY *.joblib .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]


