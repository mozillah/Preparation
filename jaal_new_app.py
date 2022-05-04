import flask
import dash
import werkzeug
from jaal import Jaal
from jaal.datasets import load_got
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
edge_df, node_df = load_got()

# app = Dash(__name__, server=server, url_base_pathname='/ATM_Data_Anlaysis/')
# app.layout = html.Div([html.H1('This Is head',style={'textAlign':'center'})])
from jaal import Jaal
from jaal.datasets import load_got

app = Jaal(edge_df, node_df).create()
dash_app = dash.Dash(__name__)
flask_app = flask.Flask(__name__)


@flask_app.route('/hello')
def hello():
    return 'Hello, World!'

application = DispatcherMiddleware(flask_app, {'/dash':app.server})


if __name__ == "__main__":
    run_simple('127.0.0.1', 5000, application,
use_reloader=True, use_debugger=True)