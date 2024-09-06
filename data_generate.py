import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate random FTIR spectral data (sampled for a few gases)
num_samples = 200
num_wavelengths = 100
wavelengths = [f'wavelength_{i+1}' for i in range(num_wavelengths)]

# Random absorbance values
data = np.random.rand(num_samples, num_wavelengths)

# Randomly assign gas labels to the samples (e.g., Gas A, Gas B, Gas C)
gases = np.random.choice(['Gas A', 'Gas B', 'Gas C'], size=num_samples)

# Create a DataFrame
df = pd.DataFrame(data, columns=wavelengths)
df['label'] = gases

# Save to CSV
#df.to_csv('ftir_data.csv', index=False)

# Plotting the FTIR spectra for a few random samples
plt.figure(figsize=(10, 6))

# Select a few random samples to plot
for i in range(5):  # Plot 5 random samples
    random_sample = np.random.randint(0, num_samples)
    plt.plot(range(1, num_wavelengths + 1), data[random_sample, :], label=f'Sample {random_sample} - {gases[random_sample]}')

plt.title('FTIR Spectra of Random Gas Samples')
plt.xlabel('Wavelength Index')
plt.ylabel('Absorbance')
plt.legend()
plt.show()
