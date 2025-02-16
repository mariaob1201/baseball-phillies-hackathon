import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import json
from functions import *
from swing import *
from io import BytesIO
import requests
import emoji
import boto3
import os

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load your CSS file
local_css("frontend/css/streamlit.css")

def us():
    # Set the title of the Streamlit application
    st.title("Home Run Scientist Team")

    # Display the logo
    logo_path = "frontend/Home_Run_Scientists.jpeg"

    # Add a description or any other content you'd like to display
    st.write("""
    ## Welcome to our Hackathon Project!
    We are the Home Run Scientist team, passionate about combining the excitement of baseball with the innovation of science.
    """)

    # Add more Streamlit components as needed
    st.write("### About Us")
    st.write("""
    We are Erica, Nicole and Maria: enthusiastic data scientist participating in a hackathon. Our project aims to leverage technology and science to support business on baseball through data.
    """)
    image = Image.open(logo_path)
    st.image(image, caption='The Logo was AI generated', use_column_width=True)


def pitch_res(singlejsonexample):
    # Check if the required keys exist in the dictionary
    if 'summary_acts' in singlejsonexample and 'pitch' in singlejsonexample['summary_acts']:
        pitch = singlejsonexample['summary_acts']['pitch']

        # Check if the required sub-keys exist
        pitch_type = pitch.get('type', '')
        pitch_result = pitch.get('result', {})
        pitch_speed = pitch.get('speed', {}).get('kph', 'Unknown')
        pitch_spin = pitch.get('spin', {}).get('rpm', 'Unknown')

        if 'Hit' in pitch_result:
            st.success(f"🏆 **Pitch Result**: {pitch_result}")
        else:
            st.success(f"**Result**: {pitch_result}")

        st.write(f"⚾ **Pitch Type**: {pitch_type} ⚾")
        st.write(f"💨 **Speed**: {pitch_speed} kph and **Spin** : {pitch_spin} rpm")
    else:
        st.write("Pitch information is not available in the provided data.")


def giff(singlejsonexample, hit):
    # Sample data (time, positions, velocities, accelerations)
    # Replace this with your actual data
    if hit is True:
        print('Ok hit')
    else:
        st.error("No Hit!")
    try:
        pitch_res(singlejsonexample)
        times = []
        posxs,posys,poszs = [],[],[]
        velsx,velsy,velsz = [],[],[]

        for x in singlejsonexample['samples_ball']:
            if 'time' in x.keys() and 'pos' in x.keys() and 'vel' in x.keys():
                times.append(x['time'])
                posxs.append(x['pos'][0])
                posys.append(x['pos'][1])
                poszs.append(x['pos'][2])
                velsx.append(x['vel'][0])
                velsy.append(x['vel'][1])
                velsz.append(x['vel'][2])

        time = np.array(times)
        pos_x = np.array(posxs)
        pos_y = np.array(posys)
        pos_z = np.array(poszs)

        # Calculate velocities (finite differences)
        vel_x = np.array(velsx)
        vel_y = np.array(velsy)
        vel_z = np.array(velsz)

        # Create DataFrame
        data = {
            'time': time,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'pos_z': pos_z,
            'vel_x': vel_x,
            'vel_y': vel_y,
            'vel_z': vel_z
        }

        df = pd.DataFrame(data)
        # Create 3D plot with Plotly
        fig = go.Figure()

        # Add scatter trace for the ball dynamics
        fig.add_trace(go.Scatter3d(
            x=df['pos_x'], y=df['pos_y'], z=df['pos_z'],
            mode='markers',
            marker=dict(size=4),
            line=dict(width=2)
        ))

        # Add animation
        frames = [go.Frame(data=[go.Scatter3d(
            x=df['pos_x'][:k + 1], y=df['pos_y'][:k + 1], z=df['pos_z'][:k + 1]
        )], name=f'frame{k}') for k in range(len(df))]

        fig.frames = frames
        fig.update_layout(updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                 'label': 'Play', 'method': 'animate'},
                {'args': [[None],
                          {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                 'label': 'Pause', 'method': 'animate'}
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }])

        # Display plot in Streamlit
        st.plotly_chart(fig)
    except Exception as e:
        st.error(e)

def giff_bat(singlejsonexample):
    if singlejsonexample is None:
        pass
    else:
        try:
            times = []
            handle_posxs, handle_posys, handle_poszs = [], [], []
            head_posxs, head_posys, head_poszs = [], [], []

            for sample in singlejsonexample['samples_bat']:
                if 'time' in sample.keys() and 'head' in sample.keys() and 'handle' in sample.keys():
                    times.append(sample['time'])
                    handle_posxs.append(sample['handle']['pos'][0])
                    handle_posys.append(sample['handle']['pos'][1])
                    handle_poszs.append(sample['handle']['pos'][2])
                    head_posxs.append(sample['head']['pos'][0])
                    head_posys.append(sample['head']['pos'][1])
                    head_poszs.append(sample['head']['pos'][2])

                    if 'event' in sample.keys():
                        if sample['event'] == 'Hit':
                            hit_time = sample['time']
                            hit_head_pos = sample['handle']['pos']
                            hit_handle_pos = sample['head']['pos']
                        else:
                            hit_time = None
                            hit_handle_pos = None
                            hit_head_pos = None

            time = np.array(times)
            handle_pos_x = np.array(handle_posxs)
            handle_pos_y = np.array(handle_posys)
            handle_pos_z = np.array(handle_poszs)
            head_pos_x = np.array(head_posxs)
            head_pos_y = np.array(head_posys)
            head_pos_z = np.array(head_poszs)

            # Create DataFrame
            data = {
                'time': time,
                'handle_pos_x': handle_pos_x,
                'handle_pos_y': handle_pos_y,
                'handle_pos_z': handle_pos_z,
                'head_pos_x': head_pos_x,
                'head_pos_y': head_pos_y,
                'head_pos_z': head_pos_z
            }

            df = pd.DataFrame(data)

            # Create 3D plot with Plotly
            fig = go.Figure()

            # Add scatter trace for the bat dynamics
            fig.add_trace(go.Scatter3d(
                x=df['handle_pos_x'], y=df['handle_pos_y'], z=df['handle_pos_z'],
                mode='markers',
                marker=dict(size=2, color='yellow'),
                name='Handle Position'
            ))

            fig.add_trace(go.Scatter3d(
                x=df['head_pos_x'], y=df['head_pos_y'], z=df['head_pos_z'],
                mode='markers',
                marker=dict(size=2, color='green'),
                name='Head Position'
            ))

            # Add hit point
            if hit_time is not None and hit_head_pos is not None and hit_handle_pos is not None:
                fig.add_trace(go.Scatter3d(
                    x=[hit_handle_pos[0]], y=[hit_handle_pos[1]], z=[hit_handle_pos[2]],
                    mode='markers',
                    marker=dict(size=6, color='red'),
                    name='Hit Handle Position'
                ))
                fig.add_trace(go.Scatter3d(
                    x=[hit_head_pos[0]], y=[hit_head_pos[1]], z=[hit_head_pos[2]],
                    mode='markers',
                    marker=dict(size=6, color='red'),
                    name='Hit Head Position'
                ))

            # Add animation frames
            frames = [
                go.Frame(data=[
                    go.Scatter3d(
                        x=df['handle_pos_x'][:k + 1], y=df['handle_pos_y'][:k + 1], z=df['handle_pos_z'][:k + 1],
                        mode='markers+lines', marker=dict(size=4, color='green')
                    ),
                    go.Scatter3d(
                        x=df['head_pos_x'][:k + 1], y=df['head_pos_y'][:k + 1], z=df['head_pos_z'][:k + 1],
                        mode='markers+lines', marker=dict(size=4, color='yellow')
                    ),
                    go.Scatter3d(
                        x=[df['handle_pos_x'][k], df['head_pos_x'][k]],
                        y=[df['handle_pos_y'][k], df['head_pos_y'][k]],
                        z=[df['handle_pos_z'][k], df['head_pos_z'][k]],
                        mode='lines', line=dict(color='black', width=2)
                    )
                ], name=f'frame{k}')
                for k in range(len(df))
            ]

            fig.frames = frames
            fig.update_layout(
                updatemenus=[{
                    'buttons': [
                        {'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                         'label': 'Play', 'method': 'animate'},
                        {'args': [[None],
                                  {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
                                   'transition': {'duration': 0}}],
                         'label': 'Pause', 'method': 'animate'}
                    ],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 87},
                    'showactive': False,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }]
            )
            # Display plot in Streamlit
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error: {e}")

def class_spped(x):
  if x>137.5490:
    return 'Excellent'
  elif x>=125.072 and x<137.5490:
    return 'Good'
  elif x>=109.219 and x<125.072:
    return 'Fair'
  elif x>=85.761 and x<109.219:
    return 'So-so'
  else:
    return 'Poor'


import math

def calculate_trajectory_feet(hit_speed_ft, launch_angle, spray_angle):
    """

      Args:
        hit_speed_ft:
        launch_angle:
        spray_angle:

      Returns:
        range and max height
    """

    launch_angle_rad = math.radians(launch_angle)
    spray_angle_rad = math.radians(spray_angle)

    g = 32.174  # pies/s^2

    v0x = hit_speed_ft * math.cos(launch_angle_rad) * math.cos(spray_angle_rad)
    v0y = hit_speed_ft * math.sin(launch_angle_rad)

    t_total = 2 * v0y / g

    range_ft = v0x * t_total

    max_height_ft = (v0y**2) / (2 * g)

    return range_ft, max_height_ft, t_total



def contact_quality(df, hit_launch_angle, hit_spray_angle, relative_distance_from_handle_to_hit,swing_velocity_head_hit, swing_displacement_handle_hit_short):

    try:
        if hit_launch_angle < 0:
            hit_launch_angle2 = 90 - int(hit_launch_angle)  # rebote
        else:
            hit_launch_angle2 = int(hit_launch_angle)

        payload = {"body":{
            'relative_distance_from_handle_to_hit': float(relative_distance_from_handle_to_hit),
            'swing_velocity_head_hit': float(swing_velocity_head_hit),
            "swing_displacement_handle_hit": float(swing_displacement_handle_hit_short),
            'hit_spray_angle': float(hit_spray_angle),
            'hit_launch_angle': float(hit_launch_angle2)
            }}

        payload_json = json.dumps(payload)
        #my apologies, I just havent time to configure this
        aws_access_key_id = st.secrets["aws_access_key_id"]
        aws_secret_access_key = st.secrets["aws_secret_access_key"]


        lambda_client = boto3.client('lambda',
                                     region_name='us-east-2',
                                     aws_access_key_id=aws_access_key_id,
                                     aws_secret_access_key=aws_secret_access_key)

        lambda_function_name = 'my_lambda_function'  # Replace with your Lambda function name

        # Call the Lambda function
        response = lambda_client.invoke(
            FunctionName=lambda_function_name,
            InvocationType='RequestResponse',  # Can also be 'Event' for asynchronous invocation
            Payload=payload_json
        )

        # Read the response
        response_payload = response['Payload'].read().decode('utf-8')
        response_data = json.loads(response_payload)

        # Imprime la respuesta
        if response_data.get("statusCode") == 200:
            body = json.loads(response_data.get("body", "{}"))
            prediction_hit_speed = [body.get("prediction")]
            classsore = body.get("class")
        else:
            st.write("Error:", response_data)

        # show the prediction
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write('#### Prediction Results')
            if classsore == 'Excellent':
                st.success(f"**{classsore} Class** 🎉👏")
                st.success(f'Predicted speed of the ball after contact: {prediction_hit_speed[0]:.2f} fts/s')

            elif classsore == 'Good':
                st.success(f"**{classsore} Class** 😊")
                st.info(f'Predicted speed of the ball after contact: {prediction_hit_speed[0]:.2f} fts/s.')

            elif classsore == 'Fair':
                st.info(f"**{classsore} Class**")
                st.info(f'Predicted speed of the ball after contact: {prediction_hit_speed[0]:.2f} fts/s')

            elif classsore == 'So-so':
                st.warning(f"**{classsore} Class** 😕")
                st.warning(f'Predicted speed of the ball after contact: {prediction_hit_speed[0]:.2f} fts/s, class {classsore}')
            elif classsore == 'Bad':
                st.error(f"**{classsore} Class** ❌")
                st.error(f'Predicted speed of the ball after contact: {prediction_hit_speed[0]:.2f}.')
            else:
                st.error('Failed to classify. Please check your input or try again later.')
        with c2:
            expected_ball(hit_launch_angle2, hit_spray_angle, prediction_hit_speed[0])

        range, maxheight, t_total = calculate_trajectory_feet(prediction_hit_speed[0], hit_launch_angle2,
                                                              hit_spray_angle)
        st.write(
            f"The expected range that the ball can takes under this hit is {range:.2f} ft and the maximum height {maxheight:.2f} ft and time flying {t_total:.2f} s.")
        st.write(
            "A negative launch angle is interpreted as a rebound or the ball and in such case the simulation takes the complementary angle (90-launch angle) to plot the trajectory as it started from the rebound.")
        # st.write(df.columns)
        st.subheader('Players Rating From Model')
    except Exception as e:
        st.error(e)


def score_player():
    import plotly.express as px
    # df = result # iris is a pandas DataFrame
    scoreevals = pd.read_csv('data/score_evaluations.csv')
    scoreevals['rank'] = scoreevals['total_weighted_score'].rank(ascending=False)
    #scoreevals['rank_average_predic'] = scoreevals['mean_prediction'].rank(ascending=True)

    opciones=sorted(list(set(scoreevals['personId'].values)))
    player_chose = st.selectbox('Choose player id:', opciones)

    # Highlight the chosen player's point
    scoreevals['color'] = ['red' if pid == player_chose else 'blue' for pid in scoreevals['personId']]

    # Create the scatter plot
    fig = px.scatter(scoreevals, x="total_weighted_score", y="rank",
                     color='color',
                     hover_data=['personId', 'mean_hit_speed', 'std_hit_speed', 'mean_prediction',
                                 'Excellent', 'Fair', 'Good', 'Poor', 'So-so',
                                 'N', 'total_weighted_score', 'scaled_score','mean_prediction','total_weighted_score'],
                     title='Ranking of predicted hit speed')

    # Update the layout to hide the legend
    fig.update_layout(showlegend=False)
    # Show the plot in Streamlit
    st.plotly_chart(fig)


    scoreresult = pd.read_csv('data/players_evaluations.csv')
    #st.write(scoreresult.columns)
    scoreresult['personID']=scoreresult['events_personId'].astype(str)
    scoreresult['hit_speed_prediction'] = scoreresult['prediction']

    colums_to_plot = ['hit_speed_prediction',
            'hit_speed_mph',
            'relative_distance_from_handle_to_hit',
            'bat_max_speed_bhit',
            'hit_spray_angle',
            'hit_launch_angle','cluster_class','prediction']
    score=scoreresult[scoreresult['events_personId']==player_chose][colums_to_plot]
    st.write(f'#### Player stats in terms of the predicted hit speed')
    st.write(f"**Rank**: {scoreevals[scoreevals['personId']==player_chose]['rank'].values[0]}")
    st.write(f"**Weighted total score**: {scoreevals[scoreevals['personId']==player_chose]['total_weighted_score'].values[0]}")

    scorecount=score.groupby('cluster_class')['prediction'].describe()
    st.dataframe(scorecount)

    #st.dataframe(score)

def extra(df):
    col1, col2, col3 = st.columns(3)
    units = 'rpm'
    with col1:
        st.markdown('**Pitch spin**')
        try:
            fig1 = explanation(df, 'pitch_spin_rpm', pitch_ball_spin_mph, units, "Pitch spin")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(e)
    # User input for second plot
    with col2:
        st.markdown('**Hit spin**')
        try:
            fig1 = explanation(df[~df['hit_spin'].isna()], 'hit_spin', hit_spin_mph, units, "Hit spin")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(e)

    df1=df[~df['hit_spin'].isna()]
    df1["spin_difference"] = df1["pitch_spin_rpm"]-df1["hit_spin"]
    with col3:
        st.markdown('**Spin diff**')
        try:
            fig1 = explanation(df1, 'spin_difference', pitch_ball_spin_mph-hit_spin_mph, units, "Pitch spin difference")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(e)

    col4, col5, col6 = st.columns(3)
    df1 = df[~df['hit_spin'].isna()]
    with col4:
        st.markdown('**Spray angle**')
        try:
            fig1 = explanation(df1, 'hit_spray_angle', hit_spray_angle, "degrees", "Pitch spin")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(e)
    # User input for second plotE
    with col5:
        st.markdown('**Launch angle**')
        try:
            fig1 = explanation(df1, 'hit_spray_angle', hit_launch_angle, "degrees", "Hit spin")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(e)

    with col6:
        st.markdown('**Spin diff**')
        try:
            fig1 = explanation(df1[~df1['relative_distance_from_handle_to_hit'].isna()], 'relative_distance_from_handle_to_hit', relative_distance_from_handle_to_hit, '',
                               "Bat hit as relative distance from handle to head")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(e)


def expected_ball(hit_launch_angle, hit_spray_angle, hit_speed_fts):
    try:
        # Constants
        g = 32.17  # Gravity constant in ft/s^2
        angle_rad = np.radians(hit_launch_angle)  # Convert to radians
        spray_angle_rad = np.radians(hit_spray_angle)  # Convert to radians
        speed_fts = hit_speed_fts  # Speed in feet per second

        # Time of flight calculation
        t_flight = (2 * speed_fts * np.sin(angle_rad)) / g
        # Time intervals
        t = np.linspace(0, t_flight, num=500)

        # Ball trajectory calculation in feet starting at 3.5 feet over the ground
        x = speed_fts * t * np.cos(angle_rad) * np.cos(spray_angle_rad)
        y = speed_fts * t * np.cos(angle_rad) * np.sin(spray_angle_rad)
        z = 3.5 + speed_fts * t * np.sin(angle_rad) - 0.5 * g * t ** 2

        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=2))])
        fig.update_layout(
            scene=dict(
                xaxis_title='X (ft)',
                yaxis_title='Y (ft)',
                zaxis_title='Z (ft)'
            ),
            title='Expected Ball Trajectory'
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(e)


def plot_histogram_with_input_line(df, val,input_value):

    mean_ = np.mean(df[val])
    std_ = np.std(df[val])

    # Sample size
    n = len(df)

    # Calculate standard error of the mean
    sem = std_ / np.sqrt(n)

    # Critical value for 95% confidence interval
    z = 1.96  # for 95% CI

    # Calculate confidence interval
    lower_ci = mean_ - z * sem
    upper_ci = mean_ + z * sem

    # Store values in a dictionary
    results = {
        'mean': mean_,
        'std': std_,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }

    if results['lower_ci'] < input_value <results['upper_ci']:
        conclusion = "The value is within the CI."
    elif results['lower_ci'] > input_value:
        conclusion = "The value is below the low CI."
    else:
        conclusion = "The value is above the upper CI."

    st.write(f"Mean: {results['mean']:.2f}")
    st.write(f"Standard Deviation: {results['std']:.2f}")
    st.write(f"95% Confidence Interval: ({results['lower_ci']:.2f}, {results['upper_ci']:.2f})")
    st.write(f"The input value {input_value:.2f} and {conclusion}")


def explanation(df, var, input_value, units, txt):
    hitdata = df[df['hit_inidicator'] == 'Hit']
    hitdata = hitdata[hitdata['hit_launch_angle']>=0]

    st.write(f"{txt} {units}")
    #(df, 'pitch_spin_rpm', pitch_ball_spin_mph, 'rpm', "Pitch spin")
    fig = px.histogram(df, x=var, nbins=30, title=f'Distribution 💨 {txt} 💨')

    # Add vertical line for the input value
    fig.add_vline(x=input_value, line_dash="dash", line_color="red")

    # Add annotations
    fig.add_annotation(
        x=input_value, y=max(np.histogram(df[var], bins=30)[0]),
        text=f"Input {units}", showarrow=True, arrowhead=1
    )
    plot_histogram_with_input_line(hitdata, var, input_value)
    return fig


if __name__ == "__main__":

    df = pd.read_csv('data/retrieved_final_0724 (1).csv')
    # Title and description
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the menu below to explore different sections.")
    section = st.sidebar.radio("Sections", ["Homerun-Scientist",

                                            "Contact Quality","Good-Swing",
                                            "Pipeline (beta)"
                                            ])
    # Overview section
    if section == "Homerun-Scientist":
        us()

    if section == "Contact Quality":
        try:
            st.title('Baseball Contact Quality Classification at hit')
            st.write("""This system analyzes baseball contact of bat and ball quality based on the given input from 
            the user. **The prediction** is the hit speed, used as a proxy for contact quality classification. Later, 
            I classify the predicted hit speed.""")

            st.sidebar.header('Enter New Data')


            hit_launch_angle = st.sidebar.slider('Hit Launch Angle ° 📐', min_value=-90.0, max_value=90.0, value=45.0,
                                                 key=3)
            hit_spray_angle = st.sidebar.slider('Hit Spray Angle °  ↔️', min_value=-50.0, max_value=50.0, value=10.0,
                                                key=4)
            relative_distance_from_handle_to_hit0 = st.sidebar.slider(
                'Bat hit position as a proportional distance from handle to head (%) :bathit:', min_value=0.0,
                max_value=100.0, value=80.0, key=5)
            swing_displacement_handle = st.sidebar.slider(
                'Swing displacement fts at handle from initial point to hit :rocket:', min_value=2.0,
                max_value=5.0, value=3.0, key=6)
            swing_velocity_head_hit = st.sidebar.slider(
                'Swing velocity at the head ft/s as the ft displaced divided by the time it takes in the swing before the hit :rocket:', min_value=5.0,
                max_value=9.0, value=7.3, key=7)
            relative_distance_from_handle_to_hit = relative_distance_from_handle_to_hit0/100

            contact_quality(df, hit_launch_angle, hit_spray_angle, relative_distance_from_handle_to_hit,swing_velocity_head_hit, swing_displacement_handle)

            st.markdown("""
                        #### The model considers four key factors:
                        * **Launch Angle (Degrees):** The angle at which the ball leaves the bat after contact. Optimal launch angles vary depending on swing mechanics and desired outcome (e.g., line drive, fly ball).
                        * **Swing Angle (Degrees):** The angle at which the bat meets the ball. A well-timed swing results in a closer alignment between the swing angle and the pitch trajectory.
                        * **Bat Position:** The relative location on the bat where the ball makes contact (closer to the handle or the barrel). Contact closer to the barrel transfers more energy to the ball.
                        * **Swing max speed:**
                        
                        **Benefits:**

                        * **Objective Assessment:** Automates contact quality evaluation, removing potential biases from human observation.
                        * **Classification of the prediction**
                        * **Data-Driven Insights:** Provides a data-backed framework for understanding hitting mechanics and their impact on performance.
                        * **Scalability:** Cloud-based deployment on AWS API Gateway allows for easy integration with other baseball analytics systems.

                        """)

            score_player()
        except Exception as e:
            print("Exception ---", e)

    if section == "Good-Swing":
        try:
            swingmain()
        except Exception as e:
            print(e)
    if section == "Pipeline (beta)":
        st.write("This isa process for the next versions, continuing the data automation")
        st.title("Bat and ball dynamics")
        smple = st.sidebar.selectbox("Choose the event: ", ['12345634_5655', '12345636_23696'])
        uploaded_file = st.sidebar.file_uploader(
            "Drag and drop your bat swing data (assumes the same structure as JSONL file here)", type="jsonl")
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8").splitlines()
            jsonl_data = read_jsonl(file_content)
            singlejsonexample = jsonl_data[0]
        else:
            file_path = f"../data/{smple}.jsonl"
            singlejsonexample = read_single_object(file_path)
            #st.write(singlejsonexample['events'], singlejsonexample['summary_acts'])

        hit = False
        evs = [x for x in singlejsonexample['samples_bat'] if 'event' in x.keys()]
        for x in evs:
            if x['event']=='Hit':
                hit = True

        giff(singlejsonexample, hit)
        giff_bat(singlejsonexample)
        logic_dynamics(singlejsonexample, df,hit)

