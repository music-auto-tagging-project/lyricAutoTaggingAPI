FROM nvidia/cuda:11.7.0-base-ubuntu20.04

COPY . /app
WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y python3-pip  && \
    pip3 install -r requirements.txt
    
EXPOSE 5000

CMD ["python3","auto_tag_api.py"]