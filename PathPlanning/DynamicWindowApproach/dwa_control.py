#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
def dwa_control(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)
    return u, trajectory
