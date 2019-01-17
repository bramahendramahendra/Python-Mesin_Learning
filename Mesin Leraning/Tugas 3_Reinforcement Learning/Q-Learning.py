# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 12:34:50 2018

@author: R.O.G
"""

import numpy as np_1301150031
import pylab as plt_1301150031

# map cell to cell, add circular cell to goal point
points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]

goal = 7

import networkx as nx_1301150031
G=nx_1301150031.Graph()
G.add_edges_from(points_list)
pos = nx_1301150031.spring_layout(G)
nx_1301150031.draw_networkx_nodes(G,pos)
nx_1301150031.draw_networkx_edges(G,pos)
nx_1301150031.draw_networkx_labels(G,pos)
plt_1301150031.show()

# how many points in graph? x points
MATRIX_SIZE = 8

# create matrix x*y
matrix_xy_1301150031 = np_1301150031.matrix(np_1301150031.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
matrix_xy_1301150031 *= -1

# assign zeros to paths and 100 to goal-reaching point
for point in points_list:
    print(point)
    if point[1] == goal:
        matrix_xy_1301150031[point] = 100
    else:
        matrix_xy_1301150031[point] = 0

    if point[0] == goal:
        matrix_xy_1301150031[point[::-1]] = 100
    else:
        # reverse of point
        matrix_xy_1301150031[point[::-1]]= 0

# add goal point round trip
matrix_xy_1301150031[goal,goal]= 100

matrix_xy_1301150031

    
Q = np_1301150031.matrix(np_1301150031.zeros([MATRIX_SIZE,MATRIX_SIZE]))

# learning parameter
gamma = 0.8

initial_state = 1

def available_actions(state):
    current_state_row = matrix_xy_1301150031[state,]
    av_act = np_1301150031.where(current_state_row >= 0)[1]
    return av_act

available_act = available_actions(initial_state) 

def sample_next_action(available_actions_range):
    next_action = int(np_1301150031.random.choice(available_act,1))
    return next_action

action = sample_next_action(available_act)

def update(current_state, action, gamma):
    
  max_index = np_1301150031.where(Q[action,] == np_1301150031.max(Q[action,]))[1]
  
  if max_index.shape[0] > 1:
      max_index = int(np_1301150031.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  
  Q[current_state, action] = matrix_xy_1301150031[current_state, action] + gamma * max_value
  #print('max_value', R[current_state, action] + gamma * max_value)
  
  if (np_1301150031.max(Q) > 0):
    return(np_1301150031.sum(Q/np_1301150031.max(Q)*100))
  else:
    return (0)
    
update(initial_state, action, gamma)

# Training
scores = []
for i in range(700):
    current_state = np_1301150031.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    score = update(current_state,action,gamma)
    scores.append(score)
    #print ('Score:', str(score))
    
print("Trained Q matrix:")
print(Q/np_1301150031.max(Q)*100)

# Testing
current_state = 0
steps = [current_state]

while current_state != 7:

    next_step_index = np_1301150031.where(Q[current_state,] == np_1301150031.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np_1301150031.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)

plt_1301150031.plot(scores)
plt_1301150031.show()