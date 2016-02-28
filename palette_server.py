# coding: utf-8
from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello ():
    return 'Hello! This is palette_server.'


if __name__ == '__main__':
    app.run(port=6000)
