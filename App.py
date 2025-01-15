from doctest import debug
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import Model
import joblib


# Initialize the Dash app
app = dash.Dash(__name__)

# load model
model = Model.load_model()

# Define layout of app
app.layout = html.Div([
    html.Div([
        html.H1("Real Estate Price Prediction", style={'text-align': 'center'}),
    
        html.Div([
            dcc.Input(id='distance_to_mrt', type='number', placeholder='Distance to MRT Station (meters)', style={'margin': '10px', 'padding': '10px', 'width': '30ch'}),
            dcc.Input(id='num_convenience_stores', type='number', placeholder='Number of Convenience Stores', style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='house_age', type='number', placeholder='House age', style={'margin': '10px', 'padding': '10px'}),
            
            # html.Button('Predict Price', id='predict_button', n_clicks=0, style={'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white'}),    
        ], style={'text-align': 'center'}),
        
        html.Div([
            html.Button('Predict Price', id='predict_button', n_clicks=0, style={'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white'}),
        ], style={'text-align': 'center'}),

        html.Div(id='prediction_output', style={'text-align': 'center', 'font-size': '20px', 'margin-top': '20px'})
    
    ], style={'width': '50%', 'margin': '0 auto', 'border': '2px solid #007BFF', 'padding': '20px', 'border-radius': '10px'})
    
])


# Define callback to update output
@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [State('distance_to_mrt', 'value'), 
     State('num_convenience_stores', 'value'),
     State('house_age', 'value')]
)

def update_output(n_clicks, distance_to_mrt, num_convenience_stores, house_age):
    if n_clicks > 0 and all(v is not None for v in [distance_to_mrt, num_convenience_stores, house_age]):
        # Prepare the feature vector
        features = pd.DataFrame([[distance_to_mrt, num_convenience_stores, house_age]], 
                                columns=['Distance to the nearest MRT station', 'Number of convenience stores', 'House age'])
        # Predict
        prediction = Model.predict(features, model)[0]
        return f'Predicted House Price of Unit Area: {prediction:.2f}'
    elif n_clicks > 0:
        return 'Please enter all values to get a prediction'
    return ''



if __name__ == '__main__':
    app.run_server(debug=True)