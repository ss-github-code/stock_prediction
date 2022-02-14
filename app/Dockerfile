FROM python:3.8

# create the app user; /home/app created by adduser
RUN addgroup app
RUN adduser --disabled-password --ingroup app app

# create the appropriate directories
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
RUN mkdir $APP_HOME/staticfiles


# set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 DJANGO_SECRET_KEY=$DJANGO_SECRET_KEY AWS_SECRET_KEY=$AWS_SECRET_KEY AWS_ACCESS_KEY=$AWS_ACCESS_KEY HOST_EXT_IP_ADDRESS=$HOST_EXT_IP_ADDRESS

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

# copy project
COPY . $APP_HOME

# chown all the files to the app user
RUN chown -R app:app $APP_HOME

# change to the app user
USER app
WORKDIR $APP_HOME
