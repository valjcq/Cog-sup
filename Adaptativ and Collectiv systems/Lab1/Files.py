from IPython.core import getipython
import sys
import subprocess
execute_in_notebook = False

if 'google.colab' in str(getipython.get_ipython()):
    execute_in_notebook = True
print("Running in",str(getipython.get_ipython()))

try:
    import google.colab
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numba'])
except ImportError:
    # Not in Google Colab, continue execution normally
    pass

# Import Libraries
import time
from datetime import datetime
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from IPython.display import HTML
from matplotlib import rc
import matplotlib as mpl

display_info = True # all other messages
display_full_stats = False # display stats of number of cultures

track_cultures_over_time = True # for every frames, store (#iterations,#cultures) in cultures_over_time
cultures_over_time = [] # Array with structure: (#iterations,#cultures)^N_frames

def set_debug(_display_info = display_info, _display_full_stats = display_full_stats, _track_cultures_over_time = track_cultures_over_time):
    global display_info,display_full_stats,track_cultures_over_time
    display_full_stats = _display_full_stats
    display_info = _display_info
    track_cultures_over_time = _track_cultures_over_time

# Set up Matplotlib to display animations in the notebook
mpl.rcParams['animation.embed_limit'] = 100  # Set the limit to 100 MB or another larger value
rc('animation', html='jshtml')

# Parameters -- DEFAULT VALUES

L = 30  # Squared lattice size
F = 3   # Number of features / traits per agent
Q = -1  # Number of trait values per feature (should be >0)

num_iterations = 20000000 # 20000000 # control parameter -- rule of thumb: > L×L×10. But should be large enough to observe stabilization.
num_frames_displayed = 100 # can be set up by hand (default: 100)
num_iterations_between_frames = int(num_iterations / num_frames_displayed) # number of interaction steps per frame
iterations = 0
im = ax = traits = None

# Precompute neighbor shifts for periodic boundary conditions
neighbor_shifts = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)]) # Von Neumann

@njit
def update_traits(traits, L, F, N_steps, neighbor_shifts):
    for _ in range(N_steps):
        # Select a random agent
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        # Select a random neighbor direction
        shift = neighbor_shifts[np.random.randint(0, 4)]
        ni = (i + shift[0]) % L
        nj = (j + shift[1]) % L

        # Compute similarity
        distance = np.mean(np.where(traits[i,j]==traits[ni,nj],0,1))
        similarity = 1.0 - distance

        # With probability S, agents interact
        if np.random.rand() < similarity:
            # Select a random trait to adjust
            k = np.random.randint(0, F)
            # Agent i,j adopts the trait from neighbor ni,nj
            traits[i, j, k] = traits[ni, nj, k]
    return traits

def update(frame):
    global traits,iterations,num_iterations_between_frames,im,ax,display_info

    if display_info == True and iterations == 0:
        print ("__________")

    if iterations < num_iterations:
        traits = update_traits(traits, L, F, num_iterations_between_frames, neighbor_shifts)
        iterations = iterations + num_iterations_between_frames
        if display_info == True and iterations%(num_iterations/10) == 0:
            #print (iterations,"/",num_iterations)
            #print (f"{int(iterations/num_iterations*100)}%")
            print (".", end="", flush=True)
            if iterations == num_iterations:
                print()
        if track_cultures_over_time == True:
            cultures_over_time.append([iterations,get_stats()[0]])

    # Update the image data
    im.set_data(traits)
    #ax.set_title(f"Iteration: {frame * N_steps}")
    ax.set_title(f"Q={Q},F={F},arena={L}x{L}\nIterations: {int(iterations)} / {num_iterations} ({int(iterations/num_iterations*100)}%)")
    return [im]

def get_stats():
    global traits

    arr_2d = traits.reshape(-1, 3)
    arr_rounded = np.round(arr_2d, decimals=8)  # Round to reduce floating-point errors
    unique_vectors, counts = np.unique(arr_rounded, axis=0, return_counts=True)
    num_unique_vectors = unique_vectors.shape[0]
    sorted_indices = np.argsort(-counts)
    sorted_vectors = unique_vectors[sorted_indices]
    sorted_counts = counts[sorted_indices]

    return num_unique_vectors,sorted_vectors,sorted_counts

def display_plot(data):
    global L

    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data')
    plt.xlabel('#cultures')
    plt.ylabel('iterations')
    plt.title(f"Q={Q},F={F},arena={L}x{L}\ncultural dissemination over time")
    #plt.legend()
    plt.ylim(0, min(L*L,Q**F))
    plt.grid(True)
    plt.show()

def run(_Q, _F = F, _L = L, _num_iterations = num_iterations, _num_frames_displayed = num_frames_displayed):
    global L,F,Q,num_iterations,num_frames_displayed,traits,im,ax,iterations,display_info,display_stats,display_graphics,cultures_over_time,num_iterations_between_frames

    start_time = time.time()

    Q = _Q
    F = _F
    L = _L
    num_iterations = _num_iterations
    num_frames_displayed = _num_frames_displayed
    num_iterations_between_frames = int(num_iterations / num_frames_displayed) # must be updated
    iterations = 0

    im = ax = traits = None

    cultures_over_time = []

    # Generate Q distinct real values between 0 and 1
    values = np.linspace(0, 1, Q, endpoint=False) + (0.5 / Q)  # Shift to avoid including 0
    traits = np.random.choice(values, size=(L, L, F))
    #print (traits)

    # Set up the plot
    fig, ax = plt.subplots()
    im = ax.imshow(traits, interpolation='nearest', animated=True)
    ax.axis('off')

    # Display information
    if display_info == True:
        print ("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        print ("Arena                  :",L,"x",L)
        print ("Number of features     :",F)
        print ("Number of trait values :",Q)
        print ("Iterations             :",num_iterations)
        print ("Display",num_frames_displayed,"frames (ie. every",num_iterations_between_frames,"iterations)")

    # Create the animation
    if track_cultures_over_time == True:
        cultures_over_time.append([iterations,get_stats()[0]])

    ani = FuncAnimation(fig, update, frames=num_frames_displayed-1, blit=True, repeat=False, interval=1)

    # Display the animation
    if execute_in_notebook == False:
        plt.show() # from terminal
    else:
        display(HTML(ani.to_jshtml())) # from a Google Colab notebook

    final_num_unique_vectors,final_sorted_vectors,final_sorted_counts = get_stats()

    if display_info == True:
        print("Number of unique cultures:", final_num_unique_vectors)
    if display_full_stats == True:
        print("List of cultures (with surface):")
        for vector, count in zip(final_sorted_vectors, final_sorted_counts):
            print("Vector:", vector, "Count:", count)

    if track_cultures_over_time == True:
        display_plot(cultures_over_time)

    end_time = time.time()
    elapsed_time = end_time - start_time
    if display_info == True:
        print(f"{elapsed_time:.1f} sec.")

    return (final_num_unique_vectors, final_sorted_vectors, final_sorted_counts, cultures_over_time)



#### #### ####

print("\n",date.today(), datetime.now().strftime("%H:%M:%S"),"GMT") # timestamp is greenwich time
print("OK.")
