import numpy as np
import matplotlib.pyplot as plt


class SSVEPReference():
    def __init__(self, timepoint_num=125, harmonic_num=5, frequencies=[8,16], sampling_rate=250, phases=[0, np.pi*3/2], is_phase_harmonic=True):
        self.timepoint_num = timepoint_num
        self.harmonic_num = harmonic_num
        self.frequencies = frequencies
        self.sampling_rate = sampling_rate
        self.phases = phases
        self.class_num = len(self.frequencies)
        self.is_phase_harmonic = is_phase_harmonic

        # assert self.harmnoic_num % 2 == 0

    
    def get_refer_allfreqs(self):
        """
        Return:
            references - matrix - (class_num, 2*Nh, timepoints): artificial templates for ssvep
        """
        references = np.zeros((self.class_num, 2 * self.harmonic_num, self.timepoint_num))
        for class_i in range(self.class_num):
            references[class_i,:,:] = self.get_refer_onefreq(self.frequencies[class_i], self.phases[class_i])
        return references
    

    def get_refer_onefreq(self, frequency, phase):
        timepoints = np.linspace(0, (self.timepoint_num-1) / self.sampling_rate, self.timepoint_num)
        y = np.array([])
        for harm_ii in range(1, self.harmonic_num+1):
            if self.is_phase_harmonic:
                refer_sin = np.sin(2*np.pi*frequency*harm_ii*timepoints + harm_ii * phase)
                refer_cos = np.cos(2*np.pi*frequency*harm_ii*timepoints + harm_ii * phase)
            else:
                refer_sin = np.sin(2*np.pi*frequency*harm_ii*timepoints + phase)
                refer_cos = np.cos(2*np.pi*frequency*harm_ii*timepoints + phase)
            y = np.append(y, [refer_sin, refer_cos])
        y = y.reshape(2*self.harmonic_num, self.timepoint_num)
        return y


def main():
    refer = SSVEPReference()
    # print(refer.get_refer_allfreqs())
    references = refer.get_refer_allfreqs()
    # Plot results
    plt.switch_backend('agg')
    class_i = 0

    timepoints = np.linspace(0, (refer.timepoint_num-1)/refer.sampling_rate, refer.timepoint_num)

    plt.subplot(4, 1, 1)
    y = references[class_i, 0, :]
    plt.plot(timepoints, y)
    plt.title("sin(f)")

    plt.subplot(4, 1, 2)
    y = references[class_i, 1, :]
    plt.plot(timepoints, y)
    plt.title("cos(f)")

    plt.subplot(4, 1, 3)
    y = references[class_i, 2, :]
    plt.plot(timepoints, y)
    plt.title("sin(2f)")

    plt.subplot(4, 1, 4)
    y = references[class_i, 3, :]
    plt.plot(timepoints, y)
    plt.title("cos(2f)")

    plt.show()
    plt.savefig("references.jpg")


if __name__ == "__main__":

    main()
