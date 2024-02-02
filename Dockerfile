FROM python:3
RUN mkdir /text-to-speech
WORKDIR /text-to-speech/
COPY . /text-to-speech/
RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt
CMD ["bash"]

