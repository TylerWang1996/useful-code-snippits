#!/usr/bin/env python
# coding: utf-8

# In[10]:


from __future__ import division

import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


from matplotlib.collections import QuadMesh
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 9)


# In[6]:


class pretty_confusion_matrix():
    """Class pretty confusion matrix takes two attributes (i) (ii) :

    Attributes:
        confusion matrix : Confusion matrix obtained from two vectors , actual class and predicted class
        plot title :  Title of the plot .
    """

    def __init__(self, confusion_matrix,plot_name):
        self.confusion_matrix = confusion_matrix
        self.plot_name = plot_name

    def get_new_fig(self,figsize=[5,5]):
        fig1 = plt.figure(figsize=figsize)
        ax1 = fig1.gca()   #Get Current Axis
        ax1.cla() # clear existing plot
        return fig1, ax1

    def configcell_text_and_colors(self,array_df, lin, col, oText, facecolors, position,
                                   font_size, string_format, show_null_values=0):
        text_add = []; 
        text_del = [];
        cell_val = array_df[lin][col]
        tot_all = array_df[-1][-1]
        per = (float(cell_val) / tot_all) * 100
        curr_column = array_df[:,col]
        ccl = len(curr_column)

        #last line  and/or last column
        if(col == (ccl - 1)) or (lin == (ccl - 1)):
            #tots and percents
            if(cell_val != 0):
                if(col == ccl - 1) and (lin == ccl - 1):
                    tot_rig = 0
                    for i in range(array_df.shape[0] - 1):
                        tot_rig += array_df[i][i]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif(col == ccl - 1):
                    tot_rig = array_df[lin][lin]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif(lin == ccl - 1):
                    tot_rig = array_df[col][col]
                    per_ok = (float(tot_rig) / cell_val) * 100
                per_err = 100 - per_ok
            else:
                per_ok = per_err = 0
            per_ok_s = ['%.1f%%'%(per_ok), '100%'] [per_ok == 100]

            #text to DEL
            text_del.append(oText)

            #text to ADD
            font_prop = fm.FontProperties(size=font_size)
            text_kwargs = dict(ha="center", va="center", gid='Total', fontproperties=font_prop)      
            lis_txt = [per_ok_s, '%.1f%%'%(per_err)]
            lis_kwa = [text_kwargs]
            dic = text_kwargs.copy(); lis_kwa.append(dic);
            lis_pos = [(oText._x, oText._y)]
            #print(len(lis_pos))
            newText = dict(x=lis_pos[0][0], y=lis_pos[0][1], text=lis_txt[0], kw=lis_kwa[0])
            text_add.append(newText)


            #set background color for sum cells (last line and last column)
            #carr = [0.27, 0.30, 0.27, 1.0]
            carr = [0.90,0.90,0.90,0.90]
            if(col == ccl - 1) and (lin == ccl - 1):
                carr = [0.90,0.90,0.90,0.90]
            facecolors[position] = carr

        else:        
            if(per > 0):
                #txt = '%s\n%.2f%%' %(cell_val, per)
                txt = '%s' %(cell_val)
            else:
                if(show_null_values == 0):
                    txt = ''
                elif(show_null_values == 1):
                    txt = '0'
                else:
                    txt = '0'

            facecolors[position] = [0.17, 0.20, 0.17, 1.0]
            oText.set_text(txt)
            oText.set_size(font_size)

            #main diagonal
            if(col == lin):
                facecolors[position] = [0.70, 0, 0.11, 1]
            else:
                None

        return text_add, text_del



    def insert_totals(self,conf_mat):
        """ insert total column and line (the last ones) """
        sum_col = []
        for col in conf_mat.columns:
            sum_col.append( conf_mat[col].sum() )
        sum_lin = []
        for item_line in conf_mat.iterrows():
            sum_lin.append( item_line[1].sum() )
            
        conf_mat['% correct'] = sum_lin
        sum_col.append(np.sum(sum_lin))
        conf_mat.loc['% correct'] = sum_col
        return conf_mat

    def pretty_plot_confusion_matrix(self,conf_mat,annotate=True, color_map="Reds", 
                                     string_format='.1f', font_size=13,
                                     line_width=0.2, color_bar=False, figsize=[5,5], 
                                     show_null_values=0, pred_val_axis='x', save = False):
        """
        params:
        confusion_matrix    dataframe (pandas) without totals
        plot_name           name of the final plot   
        annotate            print text in each cell
        color_map           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        string_format       string formatting if required during annotation
        font_size           fontsize
        line_width          linewidth
        color_bar           boolean to specify Whether to draw a colorbar
        pred_val_axis       axis chosen to show the prediction values ('x'/'col' or 'y','lin')
 
        """
        plot_name=self.plot_name
        
        if(pred_val_axis in ('col', 'x')):
            x_label = 'Predicted Outcome'
            y_label = 'Actual Outcome'
        else:
            x_label = 'Actual Outcome'
            y_label = 'Predicted Outcome'
            conf_mat = conf_mat.T

        # create "Total" column
        confusion_matrix=self.insert_totals(conf_mat)

        #this is for print always in the same window
        fig, axis = self.get_new_fig(figsize)

        # Get the heatmap with seaborn 
        heatmap_axis = sns.heatmap(confusion_matrix, annot=annotate, annot_kws={"size": font_size}, linewidths=line_width, ax=axis,
                        cbar=color_bar, cmap=color_map, linecolor='w', fmt=string_format)

        labels_x = heatmap_axis.get_xticklabels()
        labels_y = heatmap_axis.get_yticklabels()
        
        
        # Generalize this for multi-class classification (ranges from 2 to M classes)
        for classes in range(0,confusion_matrix.shape[0]):
            labels_x[classes]='class %s'%(classes)
            labels_y[classes]='class %s'%(classes)
            
        labels_x[confusion_matrix.shape[0]-1]='% correct'
        labels_y[confusion_matrix.shape[0]-1]='% correct' 
        
        #set ticklabels rotation
        heatmap_axis.set_xticklabels(labels_x, rotation = 90, fontsize = 12)
        heatmap_axis.set_yticklabels(labels_y, rotation = 0, fontsize = 12)

        # Turn off all the ticks
        for tick in heatmap_axis.xaxis.get_major_ticks():
            tick.tick1On = False
            tick.tick2On = False
        for tick in heatmap_axis.yaxis.get_major_ticks():
            tick.tick1On = False
            tick.tick2On = False

        #face colors list
        quadmesh = heatmap_axis.findobj(QuadMesh)[0]
        facecolors = quadmesh.get_facecolors()

        #iter in text elements
        array_df = np.array( confusion_matrix.to_records(index=False).tolist() )
        text_add = []
        text_del = []
        position = -1  #from left to right, bottom to top.
        
        
        for text in heatmap_axis.collections[0].axes.texts: #ax.texts:
            pos = np.array( text.get_position()) - [0.5,0.5]
            lin = int(pos[1])
            col = int(pos[0])
            position += 1
            #set text
            txt_res = self.configcell_text_and_colors(array_df=array_df, lin=lin, col=col, oText=text, facecolors=facecolors,
                                                      position=position, font_size=font_size, string_format=string_format)

            text_add.extend(txt_res[0])
            text_del.extend(txt_res[1])

        #remove the old ones
        for item in text_del:
            item.remove()
            
        #append the new ones
        for item in text_add:
            heatmap_axis.text(item['x'], item['y'], item['text'], **item['kw'])

        #titles and legends
        heatmap_axis.set_title('Confusion matrix: ' + plot_name)
        heatmap_axis.set_xlabel(x_label, fontsize=14)
        heatmap_axis.set_ylabel(y_label, fontsize=14)
        plt.tight_layout()  #set layout slim

        if save == False:
            plt.show()
        else:
            plt.savefig('Confusion matrix_' + plot_name + '.png')
            
        return plt

            
    
    def print_confusion_matrix(self, save=False):  
        confusion_matrix=self.confusion_matrix
        plot_name=self.plot_name
        confusion_matrix_df = pd.DataFrame(confusion_matrix, index=range(0,confusion_matrix.shape[0]), columns=range(0,confusion_matrix.shape[0])) 
        self.pretty_plot_confusion_matrix(confusion_matrix_df, color_map='RdBu', save=save)

