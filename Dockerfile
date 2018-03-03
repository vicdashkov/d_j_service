FROM python:3.6

ADD requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /app

ADD . ./

EXPOSE 13321
CMD ["python", "main.py"]