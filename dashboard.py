import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import os

# Sample data
actual_data = pd.read_csv('./data/test_data.csv', index_col = 0)
predicted_data_dict = {}
for predicted_data_file in os.listdir('./data/predicted_data/'):
    model_name = predicted_data_file.split('.')[0]
    predicted_data_dict[model_name] = pd.read_csv(f'./data/predicted_data/{predicted_data_file}', index_col = 0)

print(predicted_data_dict)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
model_names_dropdown_dict = {}
model_names_dropdown_list = []
for model_name in predicted_data_dict.keys():
    model_name_new = '_'.join(model_name.split('_')[3:])
    model_names_dropdown_list.append({'label': model_name_new, 'value': model_name})

variables = actual_data.columns.values.tolist()
variables_dropdown = []

print(variables)

for var in variables:
    variables_dropdown.append({'label': var, 'value': var})
app.layout = html.Div([
    html.Label('Select a model:'),
    dcc.Dropdown(
        id='dropdown-model',
        options=model_names_dropdown_list,
        value=model_names_dropdown_list[0]['value']  # Default selected value
    ),
    

    html.Label('Select a variable:'),
    dcc.Dropdown(
        id='dropdown-variable',
        options=variables_dropdown,
        value=variables[0]  # Default selected value
    ),

    dcc.Graph(id='line-plot')
])

# Define callback to update the line plot based on dropdown selections
@app.callback(
    Output('line-plot', 'figure'),
    [Input('dropdown-model', 'value'),
     Input('dropdown-variable', 'value')]
)
def update_line_plot(selected_model, selected_variable):
    print(f'selected model is {selected_model}, slected variable is {selected_variable}')
    fig = go.Figure()

    fig.add_trace(go.Scatter(
            x=actual_data.index,
            y=actual_data[selected_variable],
            marker=dict(
                color="blue"
            ),
            name='Actual',
            visible = True
            # visible=(column == default_column)

        ))

        # add line / trace 2 to figure
    fig.add_trace(go.Scatter(
        x=predicted_data_dict[selected_model].index,
        y=predicted_data_dict[selected_model].loc[:, selected_variable],
        marker=dict(
            color="red"
        ),
        name='Predicted',
        visible= True
    ))

    # if selected_variable == 'temp':
    #     variable_data = model_1_temp if selected_model == 'model_1' else model_2_temp
    # elif selected_variable == 'pressure':
    #     variable_data = model_1_pressure if selected_model == 'model_1' else model_2_pressure
    # else:
    #     # Handle other variables as needed
    #     variable_data = []
    model_short_name = '_'.join(selected_model.split('_')[3:])
   
    # fig.add_trace(go.Scatter(x=[1, 2, 3], y=variable_data, mode='lines+markers', name=f'{selected_model}_{selected_variable}'))
    fig.update_layout(
    title_text=f'{selected_variable.capitalize()} Plot for model - {model_short_name}',
    height=800

    )
    # fig.update_layout(
    #     title=f'{selected_model} {selected_variable.capitalize()} Plot',
    #     xaxis_title='Time',
    #     yaxis_title=selected_variable.capitalize(),
    # )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         step="month",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        )
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host = '0.0.0.0',use_reloader=False, port = 8054)  # Turn off reloader if inside Jupyter

