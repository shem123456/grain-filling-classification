import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv(r'SVM精度评价.csv')
    # print(df)
    name_classes = df['days'][0:12]
    values1 = df['precision'][0:12]
    values2 = df['recall'][0:12]
    # print(name_classes)
    # print(values1)
    draw_plot_func(values=values1, name_classes=name_classes, 
    plot_title="mPrecision = {0:.2f}%".format(np.nanmean(values1)*100), 
    x_label='Precision', output_path = r'SVM/Precision.png', tick_font_size = 12, plt_show = True)

    draw_plot_func(values=values2, name_classes=name_classes, 
    plot_title="mRecall = {0:.2f}%".format(np.nanmean(values2)*100), 
    x_label='Recall', output_path = r'SVM/Recall.png', tick_font_size = 12, plt_show = True)