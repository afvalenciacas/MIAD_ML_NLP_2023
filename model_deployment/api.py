#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from m09_model_deployment import predict_genre

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Predicción generos de peliculas',
    description='Predicción generos de peliculas')

ns = api.namespace('predict', 
     description='Genres Classifier')
   
parser = api.parser()

parser.add_argument(
    'year',
    type = int,
    required = True)

parser.add_argument(
    'title',
    type = str,
    required = True)

parser.add_argument(
    'plot',
    type = str,
    required = True
)

parser.add_argument(
    'rating',
    type = float,
    required = True
)

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class GenresApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_genre(args['year'], args['title'], args['plot'], args['rating'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
