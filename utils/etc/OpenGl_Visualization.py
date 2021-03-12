# Import libs
from collections import deque
from imutils.video import VideoStream
import math
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import argparse
import cv2
import imutils
import time
import pandas as pd
import os
import json
from datetime import datetime
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time

def draw_connect_traj(frame, pts):
  # loop over the set of tracked points
  for i in range(1, len(pts)):
    # if either of the tracked points are None, ignore
    # them
    if pts[i - 1] is None or pts[i] is None:
      continue
    # otherwise, compute the thickness of the line and
    # draw the connecting lines
    thickness = int(np.sqrt(args.buffer / float(i + 1)) * 2.5)
    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

  return frame

def nothing():
  pass

def get_frustum_params(size):
  frustum_edges = np.array([(0, 1),
                            (0, 2),
                            (0, 3),
                            (0, 4),
                            (1, 2),
                            (1, 4),
                            (3, 2),
                            (3, 4)])

  frustum_vertices = np.array([(0, 0, 0),     # 0
                                (-2, 2, 6),     # 1
                                (2, 2, 6),      # 2 
                                (2, -2, 6),     # 3
                                (-2, -2, 6)])   # 4 
  frustum_vertices[1:, -1] += size
  return frustum_edges, frustum_vertices

def get_plane_params():
  width = 100
  height = 60
  plane_vertices = np.array([(-width, 0, -height),   # 0
                          (+width, 0, -height),   # 1
                          (+width, 0, +height),   # 2
                          (-width, 0, +height)])  # 3
  plane_edges = np.array([(0, 1),
                          (0, 3),
                          (2, 1),
                          (2, 3)])

  return plane_edges, plane_vertices

def get_ball_params(ball_position):
  size = 10
  x = ball_position[0][0]
  y = ball_position[1][0]
  z = ball_position[2][0]
  ball_vertices = np.array([(-1, 1, 1),     # 0
                            (1, 1, 1),      # 1
                            (1, -1, 1),     # 2
                            (-1, -1, 1),    # 3
                            (-1, 1, -1),    # 4
                            (1, 1, -1),     # 5
                            (1, -1, -1),    # 6
                            (-1, -1, -1)], dtype=np.float)  # 7
  ball_vertices[:, 0] += x
  ball_vertices[:, 1] += y
  ball_vertices[:, 2] += z
  ball_edges = np.array([(0, 4),
                         (0, 1),
                         (0, 3),
                         (2, 1),
                         (2, 6),
                         (2, 3),
                         (5, 4),
                         (5, 6),
                         (5, 1),
                         (7, 3),
                         (7, 4),
                         (7, 6)])

  return ball_edges, ball_vertices

def gl_draw_function(edges, vertices, extrinsic, colors=(0, 1, 0)):
  glLineWidth(3)
  glBegin(GL_LINES)
  for edge in edges:
    for vertex in edge:
      vertex_transformed = extrinsic @ np.concatenate((vertices[vertex], [1]))
      vertex_transformed /= vertex_transformed[-1]
      glColor3fv(colors)
      glVertex3fv(vertex_transformed[:-1].reshape(-1, 1))
  glEnd()

def get_axis():
  xaxis_edges = np.array([(0, 1),])
  xaxis_vertices = np.array([(0, 0, 0),       # 0
                            (10, 0, 0),])     # 1 X-axis
  yaxis_edges = np.array([(0, 1),])
  yaxis_vertices = np.array([(0, 0, 0),       # 0
                            (0, 10, 0),])     # 1 Y-axis
  zaxis_edges = np.array([(0, 1),])
  zaxis_vertices = np.array([(0, 0, 0),       # 0
                            (0, 0, 10),])     # 1 Z-axis

  return [xaxis_edges, yaxis_edges, zaxis_edges], [xaxis_vertices, yaxis_vertices, zaxis_vertices]

def get_ball_traj_params(pts_ball_traj):
  ball_traj_vertices = np.array([(pts_ball_traj[i][0][0], pts_ball_traj[i][1][0], pts_ball_traj[i][2][0]) for i in range(len(pts_ball_traj))])
  ball_traj_edges = np.array(list(zip(range(len(pts_ball_traj)), range(len(pts_ball_traj))[1:])))
  return ball_traj_edges, ball_traj_vertices

def visualize_frustum(arucoPlaneToCameraMatrix, camera_coordinates, stereo_calibrated_dict, pts_ball_traj):
  # Real world
  # Draw each axis
  arucoCameraToPlaneMatrix = np.linalg.inv(arucoPlaneToCameraMatrix).copy()
  axis_edges, axis_vertices = get_axis()
  colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
  for i in range(len(axis_edges)):
    gl_draw_function(edges=axis_edges[i], vertices=axis_vertices[i], extrinsic=np.identity(4), colors=colors[i])
  # Draw realworld frustum
  frustum_axis_edges, frustum_vertices = get_axis()
  colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
  for i in range(len(axis_edges)):
    gl_draw_function(edges=axis_edges[i], vertices=axis_vertices[i], extrinsic=arucoCameraToPlaneMatrix, colors=colors[i])
  frustum_edges, frustum_vertices = get_frustum_params(size=2)
  gl_draw_function(edges=frustum_edges, vertices=frustum_vertices, extrinsic=arucoCameraToPlaneMatrix, colors=(1, 0, 0))
  # Draw plane of world coordinates
  plane_edges, plane_vertices = get_plane_params()
  gl_draw_function(edges=plane_edges, vertices=plane_vertices, extrinsic=np.identity(4), colors=(0, 1, 0))
  # Draw ball
  ball_edges, ball_vertices = get_ball_params(ball_position=camera_coordinates)
  gl_draw_function(edges=ball_edges, vertices=ball_vertices, extrinsic=arucoCameraToPlaneMatrix, colors=(1, 0.3, 0))
  # Draw ball trajectory
  if len(pts_ball_traj) > 1:
    ball_traj_edges, ball_traj_vertices = get_ball_traj_params(pts_ball_traj=pts_ball_traj)
    gl_draw_function(edges=ball_traj_edges, vertices=ball_traj_vertices, extrinsic=arucoCameraToPlaneMatrix, colors=(1, 1, 0))
  # Unity 
  # Use an extrinsic from real world camera to compute R|T and set it in unity
  arucoCameraToPlaneMatrix[:-1, 1] *= -1
  frustum_euler_rotation, frustum_translation = get_EulerR_Translation_from_extrinisc(extrinsic=arucoCameraToPlaneMatrix)
  # CameraToWorldMatrix came from [R|T] directly - Check weather it has the same frustum position. 
  unityCameraToWorldMatrix = get_unity_cam_extrinsic(euler_rotation=frustum_euler_rotation, translation=frustum_translation)
  # Draw Unity frustum
  unity_frustum_edges, unity_frustum_vertices = get_frustum_params(size=3)
  print("Equality of extrinsic : ", (unityCameraToWorldMatrix - arucoCameraToPlaneMatrix) < 1e-6)
  gl_draw_function(edges=unity_frustum_edges, vertices=unity_frustum_vertices, extrinsic=unityCameraToWorldMatrix, colors=(0, 0, 1))
  # Opengl coordinates (Inverted of z-coordinates)
  unityCameraToWorldMatrix[:-1, 2] *= -1
  unity_frustum_edges, unity_frustum_vertices = get_frustum_params(size=3)
  gl_draw_function(edges=unity_frustum_edges, vertices=unity_frustum_vertices, extrinsic=unityCameraToWorldMatrix, colors=(1, 1, 1))

  # Get FOV
  fov = get_fov(stereo_calibrated_dict=stereo_calibrated_dict)

  return fov, frustum_euler_rotation, frustum_translation, unityCameraToWorldMatrix, arucoCameraToPlaneMatrix



if __name__ == '__main__':
  # construct the argument parse and parse the arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--lv", dest='left_video', help="path to the (optional) video file")
  parser.add_argument("--rv", dest='right_video', help="path to the (optional) video file")
  parser.add_argument("--b", dest='buffer', type=int, default=64, help="max buffer size")
  parser.add_argument("--output_filename", dest='output_filename', type=str, help='Input the filename to save .csv file from ball tracking', required=True)
  args = parser.parse_args()

  # Initial the Pygame and some glParameters for frustum visualization
  pygame.init()
  display = (1024, 768)
  pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
  gluPerspective(30, (display[0]/display[1]), 0.1, 1000.0)  # Degree of perspective, Aspect ratio, znear, zfar)
  glTranslatef(0.0, -30.0, -300)
  glRotatef(0, 1, 0, 0)


  while True:
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    pts_ball_traj.append(camera_coordinates)
    fov, R_unity, T_unity, unityCameraToWorldMatrix, arucoCameraToPlaneMatrix = visualize_frustum(arucoPlaneToCameraMatrix, camera_coordinates, stereo_calibrated_dict, pts_ball_traj)
    pygame.display.flip()
    pygame.time.wait(10)
    # glRotatef(2, 0, 1, 0)
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

