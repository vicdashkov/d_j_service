
from sanic import Sanic
from sanic import response
from predictor import Predictor
import json
import h2o

app = Sanic()
h2o.init(ip='h2o')


@app.listener('before_server_start')
async def init(_app, _loop):
    predictor = Predictor(path_to_w2v_model='data/w2v.hex', path_to_dad_joke_model='data/dad_jokes_model.hex')
    app.predictor = predictor
    print('initialized')


@app.route('/predict', methods=['POST'])
async def predict(request):
    joke_text = request.body.decode()
    print(joke_text)
    joke_text = json.dumps(joke_text)
    prediction = request.app.predictor.make_prediction(joke_text)
    print(prediction)
    return response.json({'probability': prediction})


def main():
    app.static('/', './static')
    app.static('/', './static/index.html')
    app.run(host='0.0.0.0', port=13321)


if __name__ == '__main__':
    main()
