from flask import Flask, request, jsonify
from torch import FloatTensor
from autoencoder_train_real import *
from implement import *
from crawler import *
from implement_insta import *

app = Flask(__name__)

model =autoencoder_train()


#인스타 계정

@app.route('/instagram', methods=['POST'])
def AAA():
    id= request.json['instagramID']   #인스타 계정 (id 써서 하면됨)
    data =insta_crawler((id))
    data= insta_model(model, data)

    return jsonify(data)


@app.route('/normal', methods=['POST'])
def AiModel():
    include= request.json['include'] # 좋아하는 음식 (인덱스 리스트)
    exclude=request.json['exclude'] # 싫어하는 음식 (이름 리스트)
    data= train_model(model, include, exclude)
    return jsonify(data)

if __name__ == "__main__":
    app.run(port=5000)