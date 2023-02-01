import numpy as np
import matplotlib.pyplot as plt

# We have a set of 199 arrays each of which contain the various histogram heights calculated for a single static field
# direction, we first need to combine and re-normalise (s.t. resonance effects sum to unity) these 199 histograms

multiple_histogram_heights = np.load("alice/06_nuc_FH1'_added/all_normalised_heights.npy")
number_of_histograms_to_combine = len(multiple_histogram_heights)
histogram_heights = sum(histogram_height / number_of_histograms_to_combine for histogram_height in multiple_histogram_heights)

# histogram_heights is an array of length 2400, reflecting that we have frequency bins from 0 MHz up to 240 MHz with a
# frequency bin width of 0.1 MHz (Note: this is not as described in the paper, but it is what we have here...)

bin_centres = np.arange(0,240,0.1)

# We can thus plot Alice's original histogram

fig1,ax1 = plt.subplots()
plt.bar(bin_centres,histogram_heights,width=0.1)

# We now wish to combine the resonance effects in adjacent bins to re-plot the histogram with different sized frequency
# bins to get a better sense of how the resonance effect falls off as we approach Vmax

def Histogram_Heights_New_Bin_Width(bin_width):
    bins_combined = int(bin_width / 0.1)
    new_bin_centres = np.arange(0,240, bin_width)
    new_histogram_heights = [np.sum(histogram_heights[i*bins_combined: (i+1)*bins_combined]) for i in range(len(histogram_heights)//bins_combined)]

    fig2,ax2 = plt.subplots()
    plt.bar(new_bin_centres, new_histogram_heights, width=bin_width)

Histogram_Heights_New_Bin_Width(10)
Histogram_Heights_New_Bin_Width(5)
plt.show()