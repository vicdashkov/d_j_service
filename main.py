from sanic import Sanic
from sanic.response import json as sanci_json
from predictor import Predictor
import json
import h2o

app = Sanic()
h2o.init()


@app.listener('before_server_start')
async def init(_app, _loop):
    predictor = Predictor()
    predictor._init_w2v('/volume_data_files/w2v.hex')
    predictor._init_dad_joke_model('/volume_data_files/dad_jokes_model.hex')

    app.predictor = predictor
    print('initialized')


@app.route('/predict', methods=['POST'])
async def predict(request):
    joke_text = request.body.decode()
    joke_text = json.dumps(joke_text)
    return sanci_json({'probability': request.app.predictor.make_prediction(joke_text)})


def main():
    app.static('/static', './static')
    app.run(host='0.0.0.0', port=13321)


if __name__ == '__main__':
    main()