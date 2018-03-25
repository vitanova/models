"""
sheep and wolves
"""
import numpy as np
from random import sample, seed, uniform
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# random
seed(100)
# introduction
print("these simple codes generate the evolution of the number of sheep and wolves ")
print("on a prairie obeying the basic biological relations")

# parameters capturing the moving speed of sheep and wolf
lam_s = 2
lam_w = 3

# default initiation
chose_default = input("chosing default settings with \n" + 
    "side length of a square prairie =20 \n" +
    "number of sheep in the prairie = 50 \n" +
    "number of wolves in the prairie = 10 \n" +
    "number of periods simulating = 60 \n"
    "Y / N ? ")
if chose_default == 'Y':
    length = 20
    sheep = 50
    wolves = 10
    T = 60
else:
    # or set your own
    length = int(input("the side length of a square prairie is "))
    sheep = int(input("number of sheep in the prairie: "))
    wolves = int(input("number of wolves in the prairie: "))
    T = int(input("number of periods simulating: "))
total = sheep + wolves

# create a prairie and fill it
prairie = np.zeros((length, length))
init = sample(range(int(length*length)), total)
init_sheep = init[: sheep]
init_wolves = init[sheep: ]

# change sampled 1d index to 2d indices
def reshape(index):
    # get the row number
    i = index //length
    # get the column number
    j = index % length
    return i, j

# two functions for setting and getting values
def set_value(prairie, indices, value):
    prairie[indices[0]][indices[1]] = value 
def get_value(prairie, indices):
    return prairie[indices[0]][indices[1]]

# populate the prairie
for s in init_sheep:
    set_value(prairie, reshape(s), 1)
for w in init_wolves:
    set_value(prairie, reshape(w), -1)

def one_step_move(indices):
    i, j = indices
    # roll a dice to decide where to go
    result = np.random.uniform()
    if result < 0.25 and i - 1 >= 0:
        # upward
        potential_loc = (i - 1, j)
    elif 0.25 <= result < 0.5 and i + 1 < length:
        # downward
        potential_loc = (i + 1, j)
    elif 0.5 <= result < 0.75 and j - 1 >= 0:
        # leftward
        potential_loc = (i, j - 1)
    elif 0.75 <= result and j + 1 < length:
        # rightward
        potential_loc = (i, j + 1)
    else:
        potential_loc = (i, j)
    return potential_loc
        
def sheep_action(prairie, indices):
    # simulate the one period move of a sheep at location (i, j)
    # given the distribution of sheep and wolves in the prairie
    # it can only moves in four directions, and one step per move
    # the number of steps it can move obeys Poisson distribution
    current_loc = indices
    steps = np.random.poisson(lam_s)
    step = 0
    while step < steps:
        potential_loc = one_step_move(indices)
        # reproduce if moving to new field
        if get_value(prairie, potential_loc) == 0:
            set_value(prairie, potential_loc, 1)
            current_loc = potential_loc
        step += 1

def wolf_action(prairie, indices):
    # wolf survives by eating sheep
    # and reproduce if its health attain certain level
    # for simplicity, we avoid cannibalism here
    current_loc = indices
    health = get_value(prairie, current_loc)
    # the more starving, the more it search for food
    steps = np.random.poisson(lam_w)
    step = 0
    while step < steps:
        potential_loc = one_step_move(indices)
        # gain health after eating a sheep
        if get_value(prairie, potential_loc) == 1:
            if health == -1:
                set_value(prairie, potential_loc, -1)
                current_loc = potential_loc
            else:
                set_value(prairie, potential_loc, health + 1)
                set_value(prairie, current_loc, 0)
                current_loc = potential_loc
        step += 1

def update(prairie, n=length*length):
    # n: the number of animals who have chance to move in one period
    new_prairie = prairie.copy()
    # all wolves' health decrease by 1 at the begining of the period
    for i in range(length):
        for j in range(length):
            value = get_value(new_prairie, (i, j))
            if value < 0:
                set_value(new_prairie, (i, j), value - 1)
    # n random moves
    chosen_ones = sample(range(int(length*length)), n)
    for chosen in chosen_ones:
        indices = reshape(chosen)
        value = get_value(new_prairie, indices)
        if value > 0:
            sheep_action(new_prairie, indices)
        elif value < 0:
            wolf_action(new_prairie, indices)
    # check the health of wolves
    for i in range(length):
        for j in range(length):
            value = get_value(new_prairie, (i, j))
            if value < -10:
                set_value(new_prairie, (i, j), 0)    
    return new_prairie


current = update(prairie)
plt.rcParams['animation.convert_path'] = 'C:\Program Files\ImageMagick-7.0.7-Q8/magick.exe'
fig, ax = plt.subplots(figsize=(10, 10))
sheep_loc = []
wolves_loc = []
for i in range(length):
    for j in range(length):
        if prairie[i][j] > 0:
            sheep_loc.append([i, j])
        elif prairie[i][j] < 0:
            wolves_loc.append([i, j])
sheep_loc = np.array(sheep_loc)
wolves_loc = np.array(wolves_loc)
plt.scatter(wolves_loc[:, 0], wolves_loc[:, 1], c='black', label='wolves', marker='*')
plt.scatter(sheep_loc[:, 0], sheep_loc[:, 1], c='white', label='sheep', marker='o')
plt.axhspan(-0.7, length, facecolor='lime', alpha=0.5)
plt.axis('off')
plt.xlim(-0.6, length-0.4)
plt.ylim(-0.6, length-0.4)
plt.title('Sheep and Wolves on a Prairie')
plt.legend(bbox_to_anchor=(0, 1), loc=2)

ims = []
for t in range(T):
    before = current
    sheep_loc = []
    wolves_loc = []
    for i in range(length):
        for j in range(length):
            if before[i][j] > 0:
                sheep_loc.append([i, j])
            elif before[i][j] < 0:
                wolves_loc.append([i, j])
    sheep_loc = np.array(sheep_loc)
    wolves_loc = np.array(wolves_loc)
    img1 = plt.scatter(wolves_loc[:, 0], wolves_loc[:, 1], c='black', marker='*')
    img0 = plt.scatter(sheep_loc[:, 0], sheep_loc[:, 1], c='white', marker='o')
    txt = plt.text(10, 10, 'ROUND '+str(t+1), fontsize=20)
    ims.append([img1, img0, txt])
    current = update(before)
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)

plt.show()
#ani.save('sheep_wolves.mp4', fps=15)
input('Press ENTER to exit')
