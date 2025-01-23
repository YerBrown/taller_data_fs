# docker file for flask based api

FROM python:3.10-slim

# set working directory
WORKDIR /app

# create necessary directories
RUN mkdir -p data models

# copy files
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# train the model
RUN python train_model.py

# expose port
EXPOSE 5000

# run app with host 0.0.0.0 to allow external connections
CMD ["python", "-c", "from app import app; app.run(host='0.0.0.0', port=5000)"]

