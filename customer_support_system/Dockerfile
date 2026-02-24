# FROM python:3.10-slim
# COPY . /main
# WORKDIR /main
# RUN pip install -r requirements.txt
# CMD python main.py

FROM python:3.10-slim
WORKDIR /main
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]