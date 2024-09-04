import dash
from dash import html, dcc, Input, Output, State





# Inicializa la aplicación de Dash
app = dash.Dash(__name__)
app.title = "Equation Translator"  # Nombre de la aplicación
server = app.server
# Define el layout de la aplicación
app.layout = html.Div([
    # Encabezado con el nombre de la aplicación
    html.Header([
        html.H1("Equation Translator", style={'text-align': 'center', 'color': '#2C3E50', 'margin-bottom': '30px'})
    ], style={'background-color': '#F8F9F9', 'padding': '20px', 'border-bottom': '2px solid #D5DBDB'}),

    # Área de entrada para la secuencia
    html.Div([
        dcc.Textarea(id='input-sequence', placeholder='Enter a sequence...', style={'width': '100%', 'height': 150, 'border-radius': '5px', 'border': '1px solid #D5DBDB', 'padding': '10px'}),
        html.Button('Predict', id='predict-button', n_clicks=0, style={'margin-top': '10px', 'background-color': '#3498DB', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
        html.Div(id='output', style={'margin-top': '20px', 'font-family': 'Arial, sans-serif', 'color': '#2C3E50'})
    ], style={'width': '80%', 'margin': 'auto'}),

    # Sección About Us
    html.Div([
        html.H2("About Us", style={'text-align': 'center', 'color': '#2C3E50', 'margin-top': '50px'}),
        html.P("Equation Translator es una herramienta diseñada para traducir secuencias de texto en español a representaciones en MathML. Esta aplicación utiliza modelos de aprendizaje profundo para realizar predicciones precisas y eficaces.", style={'text-align': 'center', 'color': '#566573', 'line-height': '1.6', 'padding': '0 50px'}),
        html.P("Nuestro objetivo es facilitar la conversión de problemas matemáticos en formatos utilizables para aplicaciones web y educativas, promoviendo el uso de tecnologías avanzadas en la enseñanza y resolución de problemas matemáticos.", style={'text-align': 'center', 'color': '#566573', 'line-height': '1.6', 'padding': '0 50px'})
    ], style={'background-color': '#F8F9F9', 'padding': '50px 0', 'margin-top': '30px'})
])

# Define la función de callback para la predicción
@app.callback(
    Output('output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-sequence', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0 and value:
        prediction =" inferencia(value)"
        return html.Div([
            html.H4('Prediction:', style={'color': '#2C3E50'}),
            html.Pre(prediction, style={'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'background-color': '#F8F9F9', 'padding': '10px', 'border-radius': '5px', 'border': '1px solid #D5DBDB'})
        ])
    return ''

# Ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
        