import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from pprint import pprint
from decimal import Decimal
from matplotlib.lines import Line2D
from wordcloud import WordCloud


OUTPUT_PATH = "results/images/"
FILE_FORMAT = ".png"


def plot_graph(x_data, y_data, x_label, y_label):
    """Plots a simple line chart.

    Parameters:
    
    x_data (numpy.array): values to plot on x-axis

    y_data (numpy.array): values to plot on y-axis

    x_label (str): label fo x-axis

    y_label (str): label fo y-axis
    """
    fig, axs = plt.subplots(1, 1)

    fig.set_figheight(8)
    fig.set_figwidth(12)

    limits = get_limits_for_graph({
        "x": x_data,
        "y": y_data
    })

    axs.set_xlim([limits['min_x'], limits['max_x']])
    axs.set_ylim([limits['min_y'], limits['max_y']])

    axs.grid(True)

    axs.plot(x_data, y_data, label=y_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    graph_name = f'{x_label}'

    filepath = OUTPUT_PATH + graph_name + FILE_FORMAT

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig.savefig(filepath, format='png', bbox_inches='tight')

    plt.show()


def get_limits_for_graph(data):
    """Given a set of x and y values, returns the limits for appropriated graph plotting.

    Parameters:
    
    data (dict): dict with two keys (x and y), where each key has numpy.arrays as values

    Returns:

    dict: dict with maximum and minimum values for x and y values for adequate graph plotting
    """
    x = data['x']
    y = data['y']

    max_x = np.amax(x)
    max_y = np.amax(y)

    min_x = np.amin(x)
    min_y = np.amin(y)

    min_func = lambda x: Decimal(x) - Decimal(0.005)
    max_func = lambda x: Decimal(x) + Decimal(0.005)

    v_min_func = np.vectorize(min_func)
    v_max_func = np.vectorize(max_func)

    max_values = v_max_func([max_x, max_y])
    min_values = v_min_func([min_x, min_y])

    return  {
        "max_x": max_values[0],
        "min_x": min_values[0],
        "max_y": max_values[1],
        "min_y": min_values[1]
    }


def plot_wordcloud(text, color):
    """Creates a simple wordcloud graph using an word list.

    Parameters:
    
    text (list of str): word list to plot

    color (str): color to use, e.g. 'white', 'red', 'blue', 'green'
    """
    wordcloud = WordCloud(
        background_color="white", 
        mode="RGBA",
        max_words=10, 
        width=1800, 
        height=1400, 
        colormap='tab10',
        color_func=lambda *args, **kwargs: color).generate(" ".join(text))
    
    plt.imshow(wordcloud, interpolation='bilinear')
    
    plt.axis("off")
    plt.show()


def plot_wordcloud_by_word_probability(topic_word_prob_dist, color):
    """Creates a wordcloud graph using words sized by their probabilities.

    Parameters:
    
    topic_word_prob_dist (dict): dict whre keys are words and values are their probabilities

    color (str): color to use, e.g. 'white', 'red', 'blue', 'green'
    """
    cloud = WordCloud(background_color='white',
                    width=1800,
                    height=1400,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: color)


    cloud.generate_from_frequencies(topic_word_prob_dist, max_font_size=300)
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()


def plot_pie_chart(probabilities, labels):
    """Creates a pie chart, given a set of probabilities and its respective label.

    Parameters:
    
    probabilities (list of float): probabilities to plot on the graph

    labels (list of str): labels for each number to be plotted
    """
    fig1, ax1 = plt.subplots()
    ax1.pie(probabilities, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 16})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
