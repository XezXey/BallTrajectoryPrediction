from __future__ import print_function
# Import libs
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import os
import webbrowser
import plotly.graph_objects as go
import plotly
import tqdm

def trajectory_animation(output_xyz, gt_xyz, input_uv, lengths, n_vis, vis_idx, html_savepath='./', mask=None):
  print(output_xyz.shape, gt_xyz.shape, input_uv.shape, lengths.shape, mask.shape)
  print(n_vis, vis_idx)
  # Detach from GPU tensor object
  output_xyz = output_xyz.cpu().detach().numpy()
  gt_xyz = gt_xyz.cpu().detach().numpy()
  input_uv = input_uv.cpu().detach().numpy()
  new = 2 # Open in a new tab, if possible
  # webbrowser.open('./utils/trajectory_animation.html', new=new)
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.2)', size=3)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=3)
  plotting_string = ''
  camera = dict(
      up=dict(x=0, y=1, z=0),
      center=dict(x=0, y=0, z=0),
      eye=dict(x=0, y=1., z=1.75)
  )
  for i in tqdm.tqdm(range(n_vis), desc='Animating a trajectory...'):
    # Data for visualize
    pred_x = output_xyz[vis_idx[i]][:lengths[vis_idx[i]] + 1, 0]
    pred_y = output_xyz[vis_idx[i]][:lengths[vis_idx[i]] + 1, 1]
    pred_z = output_xyz[vis_idx[i]][:lengths[vis_idx[i]] + 1, 2]
    gt_x = gt_xyz[vis_idx[i]][:lengths[vis_idx[i]] + 1, 0]
    gt_y = gt_xyz[vis_idx[i]][:lengths[vis_idx[i]] + 1, 1]
    gt_z = gt_xyz[vis_idx[i]][:lengths[vis_idx[i]] + 1, 2]
    input_u = input_uv[vis_idx[i]][:lengths[vis_idx[i]] + 1, 0]
    input_v = input_uv[vis_idx[i]][:lengths[vis_idx[i]] + 1, 1]

    trace_xyz_pred = go.Scatter3d(x=pred_x,
                        y=pred_y,
                        z=pred_z,
                        marker=marker_dict_pred,
                        mode='markers',
                        line=dict(width=.7),
                        name='Prediction (x, y, z)')

    trace_xyz_gt = go.Scatter3d(x=gt_x,
                        y=gt_y,
                        z=gt_z,
                        marker=marker_dict_gt,
                        mode='markers',
                        line=dict(width=.7),
                        name='Ground Truth (x, y, z)')

    trace_uv = go.Scatter(x = input_u,
                        y = input_v,
                        marker=marker_dict_gt,
                        mode='markers',
                        name='Input (u, v) coordinates',
                        line=dict(width=0.7))
    frames_uv = [dict(data= [dict(type='scatter',
                               x=input_u[:k+1],
                               y=input_v[:k+1])],
                   traces=[0],  #this means that  frames[k]['data'][0]  updates trace1, and   frames[k]['data'][1], trace2 
                  )for k  in  range(1, lengths[vis_idx[i]]-1)]

    frames_xyz = [dict(data= [dict(type='scatter3d',
                               x=gt_x[:k+1],
                               y=gt_y[:k+1],
                               z=gt_z[:k+1]),
                              dict(type='scatter3d',
                               x=pred_x[:k+1],
                               y=pred_y[:k+1],
                               z=pred_z[:k+1])],
                   traces= [0, 1],  #this means that  frames[k]['data'][0]  updates trace1, and   frames[k]['data'][1], trace2 
                  )for k  in  range(1, lengths[vis_idx[i]]-1)]

    layout_xyz = go.Layout(width=768,
                       height=512,
                       scene = dict(
                          xaxis = dict(range=[-50,50], nticks=5, autorange=False),
                          yaxis = dict(range=[-2,20], nticks=5, autorange=False),
                          zaxis = dict(range=[30,-30], nticks=5, autorange=False)),
                       showlegend=True,
                       hovermode='closest',
                       scene_camera = camera,
                       updatemenus=[dict(type='buttons', showactive=False,
                                    y=-0.25,
                                    x=0.5,
                                    xanchor='right',
                                    yanchor='top',
                                    pad=dict(t=0, r=10),
                                    buttons=[dict(label='Play',
                                                  method='animate',
                                                  args=[None, dict(frame=dict(duration=0,
                                                                              redraw=True),
                                                             transition=dict(duration=0),
                                                             fromcurrent=True,
                                                             mode='immediate')])])])

    # Scene is for 3D scatter.
    # For 2D scatter use xaxis and yaxis dict-like.
    layout_uv = go.Layout(width=768,
                       height=512,
                       xaxis = dict(range=[0, 1920], autorange=False),
                       yaxis = dict(range=[0, 1080], autorange=False),
                       showlegend=True,
                       hovermode='closest',
                       updatemenus=[dict(type='buttons', showactive=False,
                                    y=-0.25,
                                    x=0.5,
                                    xanchor='right',
                                    yanchor='top',
                                    pad=dict(t=0, r=10),
                                    buttons=[dict(label='Play',
                                                  method='animate',
                                                  args=[None, dict(frame=dict(duration=0,
                                                                              redraw=True),
                                                             transition=dict(duration=0),
                                                             fromcurrent=True,
                                                             mode='immediate')])])])
    fig_uv = go.Figure(data=[trace_uv], frames=frames_uv, layout=layout_uv)
    # fig_uv.show()
    fig_xyz = go.Figure(data=[trace_xyz_gt, trace_xyz_pred], frames=frames_xyz, layout=layout_xyz)
    # fig_xyz.show()

    uvPlot = plotly.offline.plot(fig_uv,
                                config={"displayModeBar": True},
                                show_link=False,
                                include_plotlyjs=False,
                                output_type='div')

    xyzPlot = plotly.offline.plot(fig_xyz,
                                config={"displayModeBar": True},
                                show_link=False,
                                include_plotlyjs=False,
                                output_type='div')
    temp_plot_string = '''
          <div class="row" style="margin-block-start: 20px">
            <div class="col-sm-6">
              <body>
                <h2>Screen Space (u, v)</h2>
                ''' + uvPlot + '''
              </body>
            </div>
            <div class="col-sm-6">
              <body>
                <h2>World Space (x, y, z)</h2>
                ''' + xyzPlot + '''
              </body>
            </div>
          </div>
        '''
    plotting_string += temp_plot_string

  # Out of for-loop
  html_string = '''
      <html>
        <head>
          <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
          <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
          <style>body{ margin:0 100; background:whitesmoke; }</style>
          <h1> 3D Trajectory reconstruction from 2D information (u, v)</h1>
        </head>
        ''' + plotting_string + '''
      </html>'''

  with open("{}/trajectory_animation.html".format(html_savepath), 'w') as f:
      f.write(html_string)
  webbrowser.open('{}/trajectory_animation.html'.format(html_savepath), new=new)
