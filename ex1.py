import pyroomacoustics as pra
import numpy as np

# Room dimensions: 10m x 7m x 3m
room_dim = [10, 7, 3]

# Create a shoebox room
room = pra.ShoeBox(room_dim)

# Source position
source_position = [2.5, 3.5, 1.5]

# Microphone array with 3 microphones (example positions)
mic_positions = [
    [4.0, 4.0, 1.0],  # Microphone 1
    [4.1, 4.0, 1.0],  # Microphone 2
    [4.0, 4.1, 1.0]   # Microphone 3
]
mics = pra.MicrophoneArray(np.array(mic_positions).T, room.fs)

# Add the microphone array to the room
room.add_microphone_array(mics)

# Define a signal for the source (e.g., 1-second white noise signal)
fs = 16000  # Sampling frequency
signal = np.random.randn(fs)  # 1-second white noise signal

# Add source to the room with the signal
room.add_source(source_position, signal=signal)

# Compute the RIRs
room.compute_rir()

# Simulate the recording
room.simulate()


from pyroomacoustics.doa import circ_dist

# Define the DOA object
doa = pra.doa.MUSIC(mics.R, room.fs, nfft=512, c=343, num_src=1, dim=3)

# Process the microphone signals
doa.locate_sources(room.mic_array.signals)

# Get the estimated direction (azimuth and elevation)
azimuth = doa.azimuth_recon[0]
elevation = doa.elevation_recon[0]

# Convert the azimuth and elevation to Cartesian coordinates
direction = np.array([
    np.cos(elevation) * np.cos(azimuth),
    np.cos(elevation) * np.sin(azimuth),
    np.sin(elevation)
])
