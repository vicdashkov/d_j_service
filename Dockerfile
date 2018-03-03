FROM python:3.6
WORKDIR /app
ADD . ./
RUN pip install -r requirements.txt

EXPOSE 13321
CMD ["python", "main.py"]
