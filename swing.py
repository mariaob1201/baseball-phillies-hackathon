import pandas as pd
import numpy as np
# import statistics
# from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
import streamlit as st
import plotly.express as px
from functions import *


# exit_velocity =pd.read_csv('data\exit_velocity.csv')
# #exit_velocity

# exit_velocity['avg_hit_angle'].plot(kind='hist', bins=20, title='avg_hit_angle')
# plt.gca().spines[['top', 'right',]].set_visible(False)

def plot_histogram1():
    exit_velocity = pd.read_csv('data/exit_velocity.csv')
    fig = px.histogram(exit_velocity, x='avg_hit_angle', nbins=30, title=f'Average Hit Angle')

    st.plotly_chart(fig)


def plot_scatterplot():
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    exit_velocity = pd.read_csv('data/exit_velocity.csv')
    avg_hit_speed = exit_velocity['avg_hit_speed']
    avg_hit_angle = exit_velocity['avg_hit_angle']
    avg_distance = exit_velocity['avg_distance']

    st.sidebar.title('Hit Analysis')
    st.sidebar.write('Select the plot to visualize')

    plot_choice = st.sidebar.radio('Select Plot', ('Hit Speed vs Hit Distance', 'Hit Angle vs Hit Distance'))

    st.title(plot_choice)

    # Scatter plot and median lines
    if plot_choice == 'Hit Speed vs Hit Distance':
        plt.figure(figsize=(8, 6))

        # Create the scatter plot
        plt.scatter(exit_velocity['avg_hit_speed'], exit_velocity['avg_distance'])

        # Add labels and title
        plt.xlabel('avg_hit_speed')
        plt.ylabel('avg_distance')
        plt.title('Hit Speed vs Hit Distance')

        # Display the plot in Streamlit
        st.pyplot()
    elif plot_choice == 'Hit Angle vs Hit Distance':
        plt.figure(figsize=(8, 6))

        # Create the scatter plot
        plt.scatter(exit_velocity['avg_hit_angle'], exit_velocity['avg_distance'])
        plt.xlabel('avg_hit_angle')
        plt.ylabel('avg_distance')
        plt.title('Hit Angle vs Hit Distance')

        # Compute mean values
        mean_x = np.mean(exit_velocity['avg_hit_angle'])
        mean_y = np.mean(exit_velocity['avg_distance'])

        # Plot mean lines
        plt.axvline(mean_x, color='r', linestyle='--', label=f'Mean X = {mean_x:.2f}')
        plt.axhline(mean_y, color='g', linestyle='--', label=f'Mean Y = {mean_y:.2f}')
        plt.legend()

        # Show mean values
        st.write(f'Mean avg_hit_angle: {mean_x:.2f}, Mean avg_distance: {mean_y:.2f}')

        # Description of the plot
        st.write(
            'The scatter plot visualizes the relationship between average hit angle and average distance, providing insight into how the angle of a hit correlates with the distance it travels. Each point represents a hit, with its position on the x-axis indicating the average hit angle and its position on the y-axis showing the average distance achieved. The vertical red dashed line marks the mean hit angle, while the horizontal green dashed line indicates the mean distance. These mean lines help to identify central tendencies in the data, highlighting the average values for hit angle and distance. By examining the scatter plot in conjunction with these mean lines, one can better understand patterns and trends in how different hit angles affect the distance of the hit, as well as identify any deviations from the average performance.'
        )

        # Display the plot in Streamlit
        st.pyplot()


def hit_speed_scatter():
    an_data = pd.read_csv('data/retrieved_final_0724 (1).csv')
    col = an_data['hit_inidicator']  # color blue if hit, red if not hit

    # variables
    bat_average_speed_bhit = an_data['bat_average_speed_bhit']
    bat_total_displacement_bhit = an_data['bat_total_displacement_bhit']

    hit = an_data['hit_inidicator']
    colors = ['blue' if h == "Hit" else 'red' for h in hit]
    # Create the scatter plot
    plt.scatter(bat_average_speed_bhit, bat_total_displacement_bhit, c=colors)

    # Add labels and title
    plt.xlabel('bat_average_speed_bhit')
    plt.ylabel('bat_total_displacement_bhit')
    plt.title('Hit Speed vs Bat Displacement')

    # Show the plot
    st.pyplot(plt)


def swingmain():
    st.title('What Makes a Swing Good')
    st.write("This histogram shows the average hit angle")
    plot_histogram1()
    st.write("Compare Average distance and Average hit speed to Average hit angle")
    
    try:
        plot_scatterplot()
    except Exception as e:
        print(e)
    try:
        hit_speed_scatter()
    except Exception as e:
        print(e)
