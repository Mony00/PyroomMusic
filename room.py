import pyroomacoustics as pra
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# The desired reverberation time and dimensions of the room
rt60_tgt = 0.5  # reverberation time in seconds
room_dim = [9, 8, 4]  # meters

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# introducing the audio file
fs, audio = wavfile.read("bye.wav")

# Create the room
room = pra.ShoeBox(
    room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
)

# Define the sound source and add it to the room
source_position = [6, 4, 1.2]  # Coordinates adjusted to achieve a northeast direction
room.add_source(source_position, signal=audio, delay=0.5)

# Define the locations of the microphones
mic_locs = np.array([
    [6.3, 4.47, 1.2],  # Mic 1
    [6.3, 5, 1.2]     # Mic 2
]).T

# Define microphone array
mic_array = pra.MicrophoneArray(mic_locs, fs=fs)

# Place the array in the room
room.add_microphone_array(mic_array)

# Run the simulation (this will also build the RIR automatically)
room.simulate()

# Save the simulated microphone signals to a wav file
room.mic_array.to_wav("bye_room.wav", norm=True, bitdepth=np.int16)

# Calculate the cross-correlation between the received signal and the original source signal for each microphone
cross_correlations = [np.correlate(room.mic_array.signals[mic], audio, mode='full') for mic in range(room.mic_array.M)]

# Find the index of the maximum correlation for each microphone
max_correlation_indices = [np.argmax(cross_correlation) for cross_correlation in cross_correlations]

# Calculate the time delays for each microphone
time_delays = [(index - len(audio) + 1) / fs for index in max_correlation_indices]

# Print time delays for each microphone
for i, delay in enumerate(time_delays):
    print(f"Time delay for microphone {i+1}: {delay} seconds")

# Compute inter-microphone time differences (ITDs)
ITDs = np.diff(time_delays)

# Print inter-microphone time differences
for i, ITD in enumerate(ITDs):
    print(f"Inter-microphone time difference between microphone {i+1} and microphone {i+2}: {ITD} seconds")

# Speed of sound in the medium (m/s)
c = 343  # Assuming air at room temperature

# Microphone spacing
d = np.linalg.norm(mic_locs[:, 1] - mic_locs[:, 0])  # Distance between microphones

# Compute angle of arrival (DOA) for each ITD
DOAs = np.arcsin(c * ITDs / d)

# Convert angles to degrees for easier interpretation
DOAs_degrees = np.degrees(DOAs)

# Print DOAs in degrees
for i, DOA_deg in enumerate(DOAs_degrees):
    print(f"Direction of arrival for microphone pair {i+1}: {DOA_deg} degrees")

# Define tolerance for DOA mapping
epsilon = 10  # Degrees

# Map DOAs to cardinal directions
cardinal_directions = []
for DOA_deg in DOAs_degrees:
    if (0 - epsilon <= DOA_deg <= 0 + epsilon) or (360 - epsilon <= DOA_deg <= 360 + epsilon):
        cardinal_directions.append("North")
    elif 90 - epsilon <= DOA_deg <= 90 + epsilon:
        cardinal_directions.append("East")
    elif 180 - epsilon <= DOA_deg <= 180 + epsilon:
        cardinal_directions.append("South")
    elif 270 - epsilon <= DOA_deg <= 270 + epsilon:
        cardinal_directions.append("West")
    elif 0 + epsilon < DOA_deg < 90 - epsilon:
        cardinal_directions.append("Northeast")
    elif 90 + epsilon < DOA_deg < 180 - epsilon:
        cardinal_directions.append("Southeast")
    elif 180 + epsilon < DOA_deg < 270 - epsilon:
        cardinal_directions.append("Southwest")
    elif 270 + epsilon < DOA_deg < 360 - epsilon:
        cardinal_directions.append("Northwest")
    else:
        # Handle other directions or no specific direction detected
        cardinal_directions.append("Unknown")

# Print cardinal directions
for i, direction in enumerate(cardinal_directions):
    print(f"Cardinal direction for microphone pair {i+1}: {direction}")

# Plot the room with the microphones and the sound source
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot room walls
room.plot(img_order=1)
plt.title("Room Layout representing the image sources and the source propagation")

# Plot sound sources
ax.scatter(source_position[0], source_position[1], source_position[2], color='r', marker='o', label='Sound Source')

# Plot microphone array
for mic in mic_locs.T:
    ax.scatter(mic[0], mic[1], mic[2], color='b', marker='^', label='Microphone')

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sound source and microphone array representation')

# Add cardinal direction arrows
arrow_length = 1.0  # Length of arrows
arrow_style = dict(arrowstyle="->", color='g', lw=1.5)

# North (Positive Y)
ax.annotate('', xy=(0, arrow_length), xytext=(0, 0),
            arrowprops=arrow_style)
ax.text(0, arrow_length, 0, 'North', color='g')

# South (Negative Y)
ax.annotate('', xy=(0, -arrow_length), xytext=(0, 0),
            arrowprops=arrow_style)
ax.text(0, -arrow_length, 0, 'South', color='g')

# East (Positive X)
ax.annotate('', xy=(arrow_length, 0), xytext=(0, 0),
            arrowprops=arrow_style)
ax.text(arrow_length, 0, 0, 'East', color='g')

# West (Negative X)
ax.annotate('', xy=(-arrow_length, 0), xytext=(0, 0),
            arrowprops=arrow_style)
ax.text(-arrow_length, 0, 0, 'West', color='g')

# Measure the reverberation time
rt60 = room.measure_rt60()  # The program does not have a function to measure the rt30 or rt20, only the rt60
print(f"The desired RT60 was {rt60_tgt}")
print(f"The measured RT60 is {rt60[1, 0]}")

# Create a plot
figure, axs = plt.subplots(2, 2)

# Plot one of the RIRs. Both can also be plotted using room.plot_rir()
rir_1_0 = room.rir[0][0]
rir_2_0 = room.rir[1][0]

axs[0, 0].plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
axs[0, 0].set_title("The RIR from source 0 to mic 1")

# Plot signal at microphone 1
axs[1, 0].plot(room.mic_array.signals[0, :])
axs[1, 0].set_title("Microphone 1 signal")

axs[0, 1].plot(np.arange(len(rir_2_0)) / room.fs, rir_2_0)
axs[0, 1].set_title("The RIR from source 0 to mic 2")

# Plot signal at microphone 2
axs[1, 1].plot(room.mic_array.signals[1, :])
axs[1, 1].set_title("Microphone 2 signal")

for ax in axs.flat:
    ax.set(xlabel='Time [s]', ylabel='')

# Hide x labels and tick labels for top plots and y ticks for right plots
for ax in axs.flat:
    ax.label_outer()

plt.show()




# The points you're observing in figure 2 likely represent the image sources generated by Pyroomacoustics
# to simulate the room's response. These image sources are virtual sound sources created to account for
# reflections of the original sound source off the walls, floor, and ceiling of the room.
# Here's what the points typically represent:
# Exterior Points: These are the image sources located outside the room. They represent reflections of
# the original sound source off the walls and surfaces that are outside the room boundaries. These reflections
# contribute to the overall sound field captured by the microphone array.
# Interior Points: The point and cross within the room likely represent the direct sound path and the
# first-order reflection path, respectively. The point represents the direct path from the sound source to the
# microphone array, while the cross represents the first reflection off one of the room's surfaces.

# These image sources are essential for accurately modeling the room's response and capturing the effects of
# reverberation, reflections, and diffraction on the recorded microphone signals. They contribute to the
# creation of the room impulse response (RIR), which characterizes how sound propagates within the room.

# In summary, the points and crosses you're seeing in the simulation output are image sources generated by
# Pyroomacoustics to simulate the reflections and reverberation in the room environment. They are an integral
# part of the simulation process and help capture the spatial and temporal characteristics of the sound field
# within the room.






