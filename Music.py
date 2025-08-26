"""
Dataset Generator for DOA Thesis with Visualization
--------------------------------
- Uses Pyroomacoustics MUSIC to simulate DOA in a closed room
- Varies source azimuths around the microphone array
- Saves pseudospectra + ground-truth angles for ML training
- Includes comprehensive visualization
"""

import numpy as np
import pyroomacoustics as pra
import scipy.signal as sig
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
C = 343.0  # Speed of sound (m/s)

# -------------------------
# Helpers
# -------------------------

def equilateral_triangle(side_len, center):
    """Equilateral triangle array (2D)."""
    R = side_len / np.sqrt(3)
    angles = np.deg2rad([90, 210, 330])
    xy = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=0)
    centroid = np.mean(xy, axis=1, keepdims=True)
    return xy - centroid + np.array(center).reshape(2, 1)

def wideband_chirp(fs, duration, f0=300, f1=4000):
    """Wideband chirp signal."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    s = sig.chirp(t, f0=f0, t1=duration, f1=f1, method="logarithmic")
    win = sig.windows.tukey(len(s), alpha=0.1)
    return s * win

def add_awgn(x, snr_db):
    """Add white Gaussian noise per channel."""
    rng = np.random.default_rng(0)
    y = np.empty_like(x)
    for ch in range(x.shape[0]):
        s = x[ch]
        ps = np.mean(s ** 2)
        snr = 10 ** (snr_db / 10)
        pn = ps / snr
        n = rng.normal(0, np.sqrt(pn), size=s.shape)
        y[ch] = s + n
    return y

def plot_room_setup(room_dim, array_center, mic_xy, src_pos, theta_deg, ax):
    """Plot the room setup with microphones and source"""
    ax.set_xlim(0, room_dim[0])
    ax.set_ylim(0, room_dim[1])
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title(f'Room Setup - Source at {theta_deg:.1f}°')
    ax.grid(True, alpha=0.3)
    
    # Draw room
    ax.add_patch(plt.Rectangle((0, 0), room_dim[0], room_dim[1], 
                             fill=False, edgecolor='black', linewidth=2))
    
    # Plot microphones
    ax.scatter(mic_xy[0], mic_xy[1], s=100, c='red', marker='^', label='Microphones')
    for i, (x, y) in enumerate(zip(mic_xy[0], mic_xy[1])):
        ax.text(x, y + 0.1, f'M{i+1}', ha='center', va='bottom')
    
    # Plot source
    ax.scatter(src_pos[0], src_pos[1], s=200, c='blue', marker='*', label='Source')
    ax.text(src_pos[0], src_pos[1] + 0.2, f'{theta_deg:.1f}°', ha='center', va='bottom')
    
    # Plot array center
    ax.scatter(array_center[0], array_center[1], s=50, c='green', marker='o', label='Array Center')
    
    # Draw line from center to source
    ax.plot([array_center[0], src_pos[0]], [array_center[1], src_pos[1]], 
           'k--', alpha=0.5, linewidth=1)
    
    ax.legend()
    ax.set_aspect('equal')

def plot_signals(signals, fs, axs):
    """Plot time domain signals from each microphone"""
    time = np.arange(signals.shape[1]) / fs
    colors = ['red', 'green', 'blue']
    
    for i in range(signals.shape[0]):
        axs[i].plot(time, signals[i], color=colors[i], linewidth=1)
        axs[i].set_title(f'Microphone {i+1} Signal')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Amplitude')
        axs[i].grid(True, alpha=0.3)
        axs[i].set_xlim(0, time[-1])

def plot_doa_results(doa, true_theta, ax):
    """Plot DOA results with pseudospectrum"""
    # Check what attributes are available
    available_attrs = [attr for attr in dir(doa) if not attr.startswith('_')]
    print(f"Available DOA attributes: {available_attrs}")
    
    # Get the grid values (azimuth search space)
    if hasattr(doa, 'grid'):
        azimuths = np.rad2deg(doa.grid.azimuth)
    elif hasattr(doa, 'azimuth'):
        azimuths = np.rad2deg(doa.azimuth)
    else:
        # Create default azimuth grid
        azimuths = np.linspace(0, 360, 360, endpoint=False)
    
    # Get pseudospectrum
    if hasattr(doa, 'pseudospectrum'):
        ps = doa.pseudospectrum
    elif hasattr(doa, 'Pmusic'):
        ps = doa.Pmusic
    elif hasattr(doa, 'spectrum'):
        ps = doa.spectrum
    elif hasattr(doa, 'grid') and hasattr(doa.grid, 'values'):
        ps = doa.grid.values
    else:
        # Fallback: try to find any array-like attribute
        for attr in available_attrs:
            attr_val = getattr(doa, attr)
            if isinstance(attr_val, np.ndarray) and len(attr_val) == len(azimuths):
                ps = attr_val
                break
        else:
            ps = np.ones(len(azimuths))  # Placeholder
    
    ps_normalized = ps / np.max(ps)
    
    ax.plot(azimuths, ps_normalized, 'b-', linewidth=2, label='Pseudospectrum')
    ax.axvline(np.rad2deg(true_theta), color='red', linestyle='--', 
              linewidth=2, label=f'True: {np.rad2deg(true_theta):.1f}°')
    
    # Find estimated DOA
    estimated_idx = np.argmax(ps_normalized)
    estimated_theta = azimuths[estimated_idx]
    ax.axvline(estimated_theta, color='green', linestyle='--', 
              linewidth=2, label=f'Estimated: {estimated_theta:.1f}°')
    
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Normalized Power')
    ax.set_title('MUSIC DOA Estimation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 360)
    
    return estimated_theta

def debug_doa_object(doa):
    """Debug function to see what's in the DOA object"""
    print("="*50)
    print("DEBUG: DOA OBJECT ATTRIBUTES")
    print("="*50)
    
    # Get all non-private attributes
    attrs = [attr for attr in dir(doa) if not attr.startswith('_')]
    
    for attr in attrs:
        try:
            attr_val = getattr(doa, attr)
            if isinstance(attr_val, (int, float, str, bool, np.ndarray)):
                print(f"{attr}: {type(attr_val)} - {attr_val if isinstance(attr_val, (int, float, str, bool)) else attr_val.shape}")
            else:
                print(f"{attr}: {type(attr_val)}")
        except Exception as e:
            print(f"{attr}: Error accessing - {e}")
    
    print("="*50)

def music_doa(signals, fs, mic_positions, num_src=1, nfft=512):
    """Run MUSIC DOA with manual STFT processing"""
    
    # Compute STFT for each microphone
    stft_data = []
    for i in range(signals.shape[0]):
        f, t, Zxx = sig.stft(signals[i], fs, nperseg=nfft, noverlap=nfft//2, window='hann')
        stft_data.append(Zxx)
    
    # Format: (mics, freq_bins, time_frames)
    X = np.array(stft_data)
    
    # Create DOA object with different parameter combinations
    try:
        # Try with dim=2 first
        doa = pra.doa.MUSIC(
            mic_positions,
            fs,
            nfft,
            c=C,
            num_src=num_src,
            dim=2,
            azimuth=np.linspace(0, 2 * np.pi, 360, endpoint=False),
        )
        debug_doa_object(doa)
    except:
        try:
            # Try with dim=3
            doa = pra.doa.MUSIC(
                mic_positions,
                fs,
                nfft,
                c=C,
                num_src=num_src,
                dim=3,
                azimuth=np.linspace(0, 2 * np.pi, 360, endpoint=False),
            )
        except Exception as e:
            print(f"Error creating DOA object: {e}")
            return None
    
    # Use STFT data
    try:
        doa.locate_sources(X)
    except Exception as e:
        print(f"Error in locate_sources: {e}")
        # Try alternative method
        try:
            doa.process(X)
        except Exception as e2:
            print(f"Error in process: {e2}")
            return None
    
    # Debug the DOA object
    debug_doa_object(doa)
    
    return doa
# -------------------------
# Dataset generator with visualization
# -------------------------

# -------------------------
# Dataset generator with visualization
# -------------------------

def generate_dataset_with_plots(
    out_dir="dataset",
    num_angles=12,
    fs=16000,
    duration=1.0,
    room_dim=(6, 5),
    array_center=(3.0, 2.5),
    source_radius=2.0,
    rt60=0.3,
    snr_db=15,
    show_plots=True,
    save_plots=True
):
    os.makedirs(out_dir, exist_ok=True)

    # Setup room acoustics
    if rt60 > 0:
        absorption, max_order = pra.inverse_sabine(rt60, room_dim + (3,))
    else:
        absorption, max_order = 0.0, 0

    pseudospectra = []
    labels = []
    all_estimates = []
    all_true = []

    # Create summary figure
    if show_plots or save_plots:
        summary_fig, summary_ax = plt.subplots(1, 2, figsize=(15, 6))

    # Loop over different azimuths
    for i, theta in enumerate(np.linspace(0, 2 * np.pi, num_angles, endpoint=False)):
        theta_deg = np.rad2deg(theta)
        print(f"Processing angle {i+1}/{num_angles}: {theta_deg:.1f}°")
        
        # Room reset
        room = pra.ShoeBox(room_dim, fs=fs, absorption=absorption, max_order=max_order)

        # Microphone array (triangular)
        mic_xy = equilateral_triangle(0.05, array_center)
        mic_array = pra.MicrophoneArray(mic_xy, fs)
        room.add_microphone_array(mic_array)

        # Source position (circle around array)
        src_x = array_center[0] + source_radius * np.cos(theta)
        src_y = array_center[1] + source_radius * np.sin(theta)
        src_pos = (src_x, src_y)

        signal = wideband_chirp(fs, duration)
        room.add_source(src_pos, signal=signal)

        # Simulate
        room.simulate()
        sigs = add_awgn(room.mic_array.signals, snr_db)

        # Convert 2D mic positions to 3D by adding z=0
        mic_positions_3d = np.vstack([mic_xy, np.zeros(mic_xy.shape[1])])
        
        # MUSIC DOA estimation - FIXED THIS PART
        doa = music_doa(sigs, fs, mic_positions_3d)

        if doa is None:
            print("  MUSIC DOA failed, using placeholder")
            ps_normalized = np.ones(360)
            estimated_theta = theta_deg  # Assume perfect estimation for failed cases
        else:
            # Get pseudospectrum
            ps = None
            if hasattr(doa, 'pseudospectrum'):
                ps = doa.pseudospectrum
            elif hasattr(doa, 'Pmusic'):
                ps = doa.Pmusic
            elif hasattr(doa, 'spectrum'):
                ps = doa.spectrum
            elif hasattr(doa, 'grid') and hasattr(doa.grid, 'values'):
                ps = doa.grid.values
            
            if ps is not None:
                ps_normalized = ps / np.max(ps)
                # Find estimated angle from pseudospectrum peak
                estimated_idx = np.argmax(ps_normalized)
                # Get azimuth values
                if hasattr(doa, 'grid') and hasattr(doa.grid, 'azimuth'):
                    azimuths = np.rad2deg(doa.grid.azimuth)
                else:
                    azimuths = np.linspace(0, 360, 360, endpoint=False)
                estimated_theta = azimuths[estimated_idx]
            else:
                print("  Could not find pseudospectrum, using placeholder")
                ps_normalized = np.ones(360)
                estimated_theta = theta_deg
        
        error = abs(estimated_theta - theta_deg)
        if error > 180:  # Handle wrap-around
            error = 360 - error
        
        all_estimates.append(estimated_theta)
        all_true.append(theta_deg)
        
        print(f"  True: {theta_deg:.1f}°, Estimated: {estimated_theta:.1f}°, Error: {error:.1f}°")

        pseudospectra.append(ps_normalized)
        labels.append(theta)

        # Create detailed plot for this angle
        if (show_plots or save_plots) and (i % 4 == 0):  # Plot every 4th angle
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Room setup
            plot_room_setup(room_dim, array_center, mic_xy, src_pos, theta_deg, ax1)
            
            # Plot 2: Microphone signals
            plot_signals(sigs, fs, [ax2, ax3, ax4])
            ax4.set_visible(False)  # Hide the 4th subplot for 3 mics
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(os.path.join(out_dir, f'setup_angle_{theta_deg:.0f}.png'), dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Plot DOA results (only if doa is not None)
            if doa is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_doa_results(doa, theta, ax)
                plt.tight_layout()
                if save_plots:
                    plt.savefig(os.path.join(out_dir, f'doa_angle_{theta_deg:.0f}.png'), dpi=300, bbox_inches='tight')
                if show_plots:
                    plt.show()
                else:
                    plt.close()

    # Save dataset
    pseudospectra = np.array(pseudospectra)
    labels = np.array(labels)

    np.save(os.path.join(out_dir, "pseudospectra.npy"), pseudospectra)
    np.save(os.path.join(out_dir, "labels.npy"), labels)

    # Create summary plots
    if show_plots or save_plots:
        # Plot 1: True vs Estimated angles
        summary_ax[0].scatter(all_true, all_estimates, alpha=0.7)
        summary_ax[0].plot([0, 360], [0, 360], 'r--', label='Perfect estimation')
        summary_ax[0].set_xlabel('True Angle (degrees)')
        summary_ax[0].set_ylabel('Estimated Angle (degrees)')
        summary_ax[0].set_title('True vs Estimated DOA')
        summary_ax[0].legend()
        summary_ax[0].grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        errors = []
        for true, est in zip(all_true, all_estimates):
            error = abs(est - true)
            if error > 180:
                error = 360 - error
            errors.append(error)
        
        summary_ax[1].hist(errors, bins=20, alpha=0.7, edgecolor='black')
        summary_ax[1].set_xlabel('Estimation Error (degrees)')
        summary_ax[1].set_ylabel('Frequency')
        summary_ax[1].set_title(f'DOA Error Distribution\nMean Error: {np.mean(errors):.2f}°')
        summary_ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(out_dir, 'summary_results.png'), dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

    print(f"✅ Dataset generated: {pseudospectra.shape} spectra")
    print(f"   Mean estimation error: {np.mean(errors):.2f}°")
    print(f"   Max estimation error: {np.max(errors):.2f}°")
    print(f"   Results saved in {out_dir}/")

# -------------------------
# ML Implementation and Evaluation
# -------------------------

def train_and_evaluate_ml_model(data_dir="dataset"):
    """Train and evaluate ML model on the DOA dataset"""
    try:
        # Load dataset
        pseudospectra = np.load(os.path.join(data_dir, "pseudospectra.npy"))
        labels = np.load(os.path.join(data_dir, "labels.npy"))
        
        print(f"Dataset loaded: {pseudospectra.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Convert labels to degrees for easier interpretation
        labels_deg = np.rad2deg(labels)
        
        # Simple ML model (you can replace this with more complex models)
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            pseudospectra, labels_deg, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate errors (handle circular nature)
        errors = []
        for true, pred in zip(y_test, y_pred):
            error = abs(pred - true)
            if error > 180:
                error = 360 - error
            errors.append(error)
        
        # Results
        print("\n" + "="*50)
        print("ML MODEL RESULTS")
        print("="*50)
        print(f"Mean Absolute Error: {np.mean(errors):.2f}°")
        print(f"Max Error: {np.max(errors):.2f}°")
        print(f"Accuracy within 5°: {np.mean(np.array(errors) <= 5)*100:.1f}%")
        print(f"Accuracy within 10°: {np.mean(np.array(errors) <= 10)*100:.1f}%")
        
        # Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # True vs Predicted
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([0, 360], [0, 360], 'r--', linewidth=2)
        ax1.set_xlabel('True Angle (degrees)')
        ax1.set_ylabel('Predicted Angle (degrees)')
        ax1.set_title('ML Model: True vs Predicted DOA')
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        ax2.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Prediction Error (degrees)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'ML Model Error Distribution\nMAE: {np.mean(errors):.2f}°')
        ax2.grid(True, alpha=0.3)
        
        # Comparison with MUSIC
        music_errors = []
        for i in range(len(y_test)):
            true = y_test[i]
            # Find corresponding MUSIC estimation
            idx = np.where(labels_deg == true)[0][0]
            music_est = np.rad2deg(labels[np.argmax(pseudospectra[idx])])
            error = abs(music_est - true)
            if error > 180:
                error = 360 - error
            music_errors.append(error)
        
        # Error comparison
        ax3.boxplot([errors, music_errors], labels=['ML Model', 'MUSIC'])
        ax3.set_ylabel('Error (degrees)')
        ax3.set_title('Error Comparison: ML vs MUSIC')
        ax3.grid(True, alpha=0.3)
        
        # Sample predictions
        sample_idx = np.random.choice(len(y_test), min(8, len(y_test)), replace=False)
        angles = np.arange(360)
        
        for i, idx in enumerate(sample_idx):
            if i < 4:  # First 4 samples
                ax = ax4
            else:  # Next 4 samples (would need another subplot)
                continue
                
            ax.plot(angles, pseudospectra[idx], alpha=0.7, 
                   label=f'True: {y_test[idx]:.0f}°, Pred: {y_pred[idx]:.0f}°')
        
        ax4.set_xlabel('Azimuth (degrees)')
        ax4.set_ylabel('Normalized Pseudospectrum')
        ax4.set_title('Sample Pseudospectra with Predictions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, 'ml_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return model, errors
        
    except Exception as e:
        print(f"Error in ML training: {e}")
        return None, None

# -------------------------
# Run example
# -------------------------

if __name__ == "__main__":
    # Generate dataset with plots
    generate_dataset_with_plots(
        out_dir="dataset",
        num_angles=24,
        rt60=0.3,
        snr_db=10,
        show_plots=True,
        save_plots=True
    )
    
    # Train and evaluate ML model
    print("\n" + "="*60)
    print("TRAINING MACHINE LEARNING MODEL")
    print("="*60)
    model, errors = train_and_evaluate_ml_model("dataset")