# Create a figure canvas and plot the original, noisy data
def ma(y, window):
	avg_mask = np.ones(window) / window
	return np.convolve(y, avg_mask, 'same')
	

fig, ax = plt.subplots()
ax.plot(x, y, label="Original")
# Compute moving averages using different window sizes
window_lst = [3, 6, 10, 16, 22, 35]
y_avg = np.zeros((len(window_lst) , N))
for i, window in enumerate(window_lst):
	avg_mask = np.ones(window) / window
	y_avg[i, :] = np.convolve(y, avg_mask, 'same')
	# Plot each running average with an offset of 50
	# in order to be able to distinguish them
	ax.plot(x, y_avg[i, :] + (i+1)*50, label=window)
# Add legend to plot
ax.legend()
plt.show()