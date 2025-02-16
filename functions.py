import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from plotly.subplots import make_subplots
import plotly.express as px
import json



def compute_velocity(positions, times):
    velocities = np.gradient(positions, times, axis=0)
    return velocities

def compute_acceleration(velocities, times):
    accelerations = np.gradient(velocities, times, axis=0)
    return accelerations


def simplifying_positions(singlejsonexample):
    if 'time' in singlejsonexample['samples_bat'][0]:
        t_bat = [x['time'] for x in singlejsonexample['samples_bat']]
        P_head = np.array([x['head']['pos'] for x in singlejsonexample['samples_bat']])
        P_hands = np.array([x['handle']['pos'] for x in singlejsonexample['samples_bat']])
        events = [x for x in singlejsonexample['samples_bat'] if 'event' in x.keys()]
        hit_pre_info = [x for x in events if x['event'] == 'Hit' or x['event'] == 'Nearest']
        hit_info = hit_pre_info[0]
        if len(hit_pre_info)>0:
            if hit_info['event'] == 'Hit':
                st.write(f"Hit!")
            else:
                st.write(f"That was close to a hit!")

    else:
        t_bat = []
        P_head = []
        P_hands = []
        events = []
        hit_info = None

    if len(singlejsonexample['samples_ball']) > 0:
        samples =singlejsonexample['samples_ball']
        t_ball = [x['time'] for x in samples]
        ball_pos = np.array([x['pos'] for x in samples])

        # Initialize ball_vel with None
        ball_vel = [x['vel'] if 'vel' in x else None for x in samples]

        # Identify indices where velocity is missing
        missing_vel_indices = [i for i, vel in enumerate(ball_vel) if vel is None]

        if missing_vel_indices:
            # Compute velocities for missing entries
            computed_velocities = compute_velocity(ball_pos, t_ball)

            for i in missing_vel_indices:
                ball_vel[i] = computed_velocities[i]

        # Convert ball_vel to a numpy array
        ball_vel = np.array(ball_vel)
        ball_acc = [x['acc'] if 'acc' in x else None for x in samples]
        missing_acc_indices = [i for i, acc in enumerate(ball_acc) if acc is None]

        if missing_acc_indices:
            # Compute accelerations for missing entries
            computed_accelerations = compute_acceleration(ball_vel, t_ball)

            for i in missing_acc_indices:
                ball_acc[i] = computed_accelerations[i]

        # Convert ball_acc to a numpy array
        ball_acc = np.array(ball_acc)

    return t_bat, P_head, P_hands, t_ball, ball_pos, ball_vel, ball_acc, events, hit_info

# Streamlit app
def plot_trajectories(singlejsonexample):
    t_bat, P_head, P_hands, t_ball, ball_pos, ball_vel, ball_acc, events, hit_info = simplifying_positions(singlejsonexample)

    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}],
                               [{'type': 'xy'}, {'type': 'xy'}]],
                        subplot_titles=('3D Trajectories', 'Velocities', 'Bat Angle', 'Ball Acceleration'))

    # Plot 1: 3D Trajectories
    if len(P_head)==0:
        pass
    else:
        fig.add_trace(go.Scatter3d(x=P_head[:, 0], y=P_head[:, 1], z=P_head[:, 2], mode='lines', name='Bat Head'), row=1,
                      col=1)
        fig.add_trace(go.Scatter3d(x=P_hands[:, 0], y=P_hands[:, 1], z=P_hands[:, 2], mode='lines', name='Bat Handle'),
                      row=1, col=1)
    fig.add_trace(go.Scatter3d(x=ball_pos[:, 0], y=ball_pos[:, 1], z=ball_pos[:, 2], mode='lines', name='Ball'), row=1,
                  col=1)
    #st.write(hit_info)
    if hit_info is None:
        pass
    else:
        fig.add_trace(
            go.Scatter3d(x=[hit_info['head']['pos'][0]], y=[hit_info['head']['pos'][1]], z=[hit_info['head']['pos'][2]],
                         mode='markers', marker=dict(size=5, color='red'), name='Impact Point (Head)'), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=[hit_info['handle']['pos'][0]], y=[hit_info['handle']['pos'][1]],
                                   z=[hit_info['handle']['pos'][2]], mode='markers', marker=dict(size=5, color='green'),
                                   name='Impact Point (Handle)'), row=1, col=1)

    # Plot 2: Velocities
    if len(P_head)>0:
        bat_vel = np.linalg.norm(np.diff(P_head, axis=0), axis=1) / np.diff(t_bat)
        bat_handle_vel = np.linalg.norm(np.diff(P_hands, axis=0), axis=1) / np.diff(t_bat)

        fig.add_trace(go.Scatter(x=t_bat[1:], y=bat_vel, mode='lines', name='Bat head Velocity'), row=1, col=2)
        fig.add_trace(go.Scatter(x=t_bat[1:], y=bat_handle_vel, mode='lines', name='Bat handle Velocity'), row=1, col=2)
    else:
        pass

    ball_speed = np.linalg.norm(ball_vel, axis=1)
    fig.add_trace(go.Scatter(x=t_ball, y=ball_speed, mode='lines', name='Ball Speed'), row=1, col=2)
    #fig.add_vline(x=float(hit_contact_t), line=dict(color='red', dash='dash'), row=1, col=2)

    # Plot 3: Bat Angle
    if len(P_hands)>0:
        bat_angles = np.arctan2(P_head[:, 1] - P_hands[:, 1], P_head[:, 0] - P_hands[:, 0])
        fig.add_trace(go.Scatter(x=t_bat, y=np.degrees(bat_angles), mode='lines', name='Bat Angle'), row=2, col=1)
        #fig.add_vline(x=hit_contact_t, line=dict(color='red', dash='dash'), row=2, col=1)

    # Plot 4: Ball Acceleration
    ball_acc_mag = np.linalg.norm(ball_acc, axis=1)
    fig.add_trace(go.Scatter(x=t_ball, y=ball_acc_mag, mode='lines', name='Ball Acceleration'), row=2, col=2)
    #fig.add_vline(x=hit_contact_t, line=dict(color='red', dash='dash'), row=2, col=2)

    fig.update_layout(height=800, width=1000, title_text="Bat and Ball Dynamics")

    st.plotly_chart(fig)

def hit_stats(df):
    hits = df[df['bat_hit_event_category']==1]
    st.write("Hit speed")
    st.dataframe(hits['hit_speed_mph'].describe())


def plot_histogram_with_input_line(df, input_value, val, units):
    fig = px.histogram(df, x=val, nbins=30, title=f'{val} Distribution')

    # Add vertical line for the input value
    fig.add_vline(x=input_value, line_dash="dash", line_color="red")

    # Add annotations
    fig.add_annotation(
        x=input_value, y=max(np.histogram(df[val], bins=30)[0]),
        text=f"Input Value: {input_value} {units}", showarrow=True, arrowhead=1
    )

    st.plotly_chart(fig)

    mean_hit_speed = np.mean(df[val])
    std_hit_speed = np.std(df[val])

    if input_value < mean_hit_speed - std_hit_speed:
        conclusion = "The value is below average."
    elif input_value > mean_hit_speed + std_hit_speed:
        conclusion = "The value is above average."
    else:
        conclusion = "The value is around average."

    st.write(f"**Conclusion:** {conclusion}")
    st.write(f"Mean: {mean_hit_speed:.2f} mph")
    st.write(f"Standard deviation: {std_hit_speed:.2f} mph")


def display_pitch_info(df, data):
    """

    :param df:
    :param data:
    :return: ['summary_acts']['hit'] data such as spin, speed and an histogram of the hit
    """
    try:
        hitdata=df[df['hit_inidicator']=='Hit']
        sacts_hit = data['summary_acts']['hit']
        st.write(f"**Hit Details**")
        st.write(f"💨 Hit speed: {sacts_hit['speed']['mph']} mph")
        plot_histogram_with_input_line(hitdata, sacts_hit['speed']['mph'],'hit_speed_mph', 'mph')
        st.write(f"🌀 Hit spin: {sacts_hit['spin']['rpm']} rpm")
        plot_histogram_with_input_line(hitdata, sacts_hit['spin']['rpm'], 'hit_spin','rpm')
    except Exception as e:
        st.error('No hit information available of the hit ------', e)

    st.write(f"**Pitch Details**")
    sacts_pitch = data['summary_acts']['pitch']
    st.write(f"⚾ Pitch result: {sacts_pitch['result']}")
    st.write(f"⚡ Pitch speed: {sacts_pitch['speed']['mph']} mph")
    plot_histogram_with_input_line(df[df['pitch_speed_mph']>=0], sacts_pitch['speed']['mph'], 'pitch_speed_mph', 'mph')
    st.write(f"🌀 Pitch spin: {sacts_pitch['spin']['rpm']} rpm")
    plot_histogram_with_input_line(df[df['pitch_spin_rpm']>=0], sacts_pitch['spin']['rpm'], 'pitch_spin_rpm', 'rpm')


def read_csv_from_zip(zip_file_path, csv_file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(csv_file_name) as f:
            df = pd.read_csv(f)
    return df


def read_jsonl(file):
    """
    Reads a JSONL file and returns a list of dictionaries.
    """
    data = []
    for line in file:
        data.append(json.loads(line))
    return data

def read_single_object(file_path):
    # Open the file and read each line
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each JSON string into a Python dictionary
            json_obj = json.loads(line)
            # dataexample.append(json_obj)
    return json_obj

def ball_metrics(object):
    return 1,1

def bat_metrics(object):
    return 1,1

def metrics_before_hit(object):
    ball_exit_v, bat_spin = ball_metrics(object)
    bat_swing, bat_spin = bat_metrics(object)
    score = model_score(ball_exit_v, ball_spin, bat_swing, bat_spin)
    return score



def model2(score):
    if score > 600:
        # Title with emojis
        st.title('Celebration Time! 🥳🎉 You did it! 🌟🏆')

    elif score >400:
        # Using emojis in write statements
        st.write('Keep up the great work! 🌈🎶')
    else:
        st.write('Keep up the great work! 🌈🎶')

def score_classification(object):
    ball_exit_v, ball_spin = ball_metrics(object)
    bat_swing, bat_angle = bat_metrics(object)
    score = 600

    return {'score': score}


def logic_dynamics(singlejsonexample, df,hit):
    plot_trajectories(singlejsonexample)  # ok
    if hit is True:
        scores = score_classification(None)
        st.success(f"The contact quality score {scores['score']:.0f}")
        model2(scores['score']) #not mock

        display_pitch_info(df, singlejsonexample)
