FROM python:3.12.10-slim

# Create and switch to the app directory
WORKDIR /cucinabot

# Copy only requirements
COPY requirements.txt ./

# Install requirements
RUN pip install -r requirements.txt

# Now copy your entire source code
COPY . ./

# Finally, run server
CMD ["python", "app.py"]