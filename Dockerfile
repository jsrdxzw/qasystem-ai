FROM python:3.8-slim
MAINTAINER qa system ai api powered by tensorflow and flask "jsrdxzw@163.com"
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y python3-dev gcc libc-dev libffi-dev build-essential
RUN pip3 install -r requirements.txt
EXPOSE 5000
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]