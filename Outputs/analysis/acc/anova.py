import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Generate some example data
np.random.seed(0)
subjects = np.arange(1, 36)
methods = ['Method1', 'Method2', 'Method3']
time_windows = [f'TW{i}' for i in range(1, 11)]

# Create a dataframe to hold the data
data = []
for subject in subjects:
    for method in methods:
        for time_window in time_windows:
            result = np.random.rand()  # Replace with actual data
            data.append([subject, method, time_window, result])

df = pd.DataFrame(data, columns=['Subject', 'Method', 'TimeWindow', 'Result'])

# Perform two-way repeated measures ANOVA
aovrm = AnovaRM(df, 'Result', 'Subject', within=['Method', 'TimeWindow'])
res = aovrm.fit()
print(res)

# Perform post hoc tests
# For Method
tukey_method = pairwise_tukeyhsd(endog=df['Result'], groups=df['Method'], alpha=0.05)
print("\nPost Hoc Test for Method:")
print(tukey_method)

# For TimeWindow
tukey_timewindow = pairwise_tukeyhsd(endog=df['Result'], groups=df['TimeWindow'], alpha=0.05)
print("\nPost Hoc Test for TimeWindow:")
print(tukey_timewindow)

# If you also need the interaction effect post hoc tests:
# Creating a combined group factor for interaction
df['Method_TimeWindow'] = df['Method'] + "_" + df['TimeWindow']
tukey_interaction = pairwise_tukeyhsd(endog=df['Result'], groups=df['Method_TimeWindow'], alpha=0.05)
print("\nPost Hoc Test for Interaction (Method_TimeWindow):")
print(tukey_interaction)