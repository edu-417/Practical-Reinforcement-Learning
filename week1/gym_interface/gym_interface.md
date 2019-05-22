

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# This code creates a virtual display to draw game images on. 
# If you are running locally, just ignore it
import os
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
    !bash ../xvfb start
    %env DISPLAY=:1
```

    Starting virtual X frame buffer: Xvfb.
    env: DISPLAY=:1


### OpenAI Gym

We're gonna spend several next weeks learning algorithms that solve decision processes. We are then in need of some interesting decision problems to test our algorithms.

That's where OpenAI gym comes into play. It's a python library that wraps many classical decision problems including robot control, videogames and board games.

So here's how it works:


```python
import gym
env = gym.make("MountainCar-v0")

plt.imshow(env.render('rgb_array'))
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
```

    Observation space: Box(2,)
    Action space: Discrete(3)



![png](output_2_1.png)


Note: if you're running this on your local machine, you'll see a window pop up with the image above. Don't close it, just alt-tab away.

### Gym interface

The three main methods of an environment are
* __reset()__ - reset environment to initial state, _return first observation_
* __render()__ - show current environment state (a more colorful version :) )
* __step(a)__ - commit action __a__ and return (new observation, reward, is done, info)
 * _new observation_ - an observation right after commiting the action __a__
 * _reward_ - a number representing your reward for commiting action __a__
 * _is done_ - True if the MDP has just finished, False if still in progress
 * _info_ - some auxilary stuff about what just happened. Ignore it ~~for now~~.


```python
obs0 = env.reset()
print("initial observation code:", obs0)

# Note: in MountainCar, observation is just two numbers: car position and velocity
```

    initial observation code: [-0.59398965  0.        ]



```python
print("taking action 2 (right)")
new_obs, reward, is_done, _ = env.step(2)

print("new observation code:", new_obs)
print("reward:", reward)
print("is game over?:", is_done)

# Note: as you can see, the car has moved to the riht slightly (around 0.0005)
```

    taking action 2 (right)
    new observation code: [-0.59246563  0.00152402]
    reward: -1.0
    is game over?: False


### Play with it

Below is the code that drives the car to the right. 

However, it doesn't reach the flag at the far right due to gravity. 

__Your task__ is to fix it. Find a strategy that reaches the flag. 

You're not required to build any sophisticated algorithms for now, feel free to hard-code :)

_Hint: your action at each step should depend either on __t__ or on __s__._


```python

# create env manually to set time limit. Please don't change this.
TIME_LIMIT = 250
env = gym.wrappers.TimeLimit(gym.envs.classic_control.MountainCarEnv(),
                             max_episode_steps=TIME_LIMIT + 1)
s = env.reset()
actions = {'left': 0, 'stop': 1, 'right': 2}

# prepare "display"
%matplotlib notebook
fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()

def policy(s, t):
    # YOUR CODE HERE
    return  2 -  2 * ( s[1] < 1e-6 )


for t in range(TIME_LIMIT):
    
    s, r, done, _ = env.step(policy(s, t))
    
    #draw game image on display
    ax.clear()
    ax.imshow(env.render('rgb_array'))
    fig.canvas.draw()
    
    if done:
        print("Well done!")
        break
else:    
    print("Time limit exceeded. Try again.")
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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" width="432">


    Well done!


### Submit to coursera


```python
from submit import submit_interface
submit_interface(policy, <email>, <token>)
```

    Submitted to Coursera platform. See results on assignment page!



```python

```