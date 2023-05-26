#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from m09_model_deployment import predict_genre

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Predicción de géneros de películas',
    description='API para predecir los géneros de una película')

ns = api.namespace('predict', 
     description='Clasificador de géneros')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class GenresApi(Resource):

    @api.expect(api.parser().add_argument('year', type=int, required=True))
    @api.expect(api.parser().add_argument('title', type=str, required=True))
    @api.expect(api.parser().add_argument('plot', type=str, required=True))
    @api.expect(api.parser().add_argument('rating', type=float, required=True))
    @api.marshal_with(resource_fields)
    def post(self):
        args = api.payload
        year = args['year']
        title = args['title']
        plot = args['plot']
        rating = args['rating']
        
        return {
            "result": predict_genres(year, title, plot, rating)
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
