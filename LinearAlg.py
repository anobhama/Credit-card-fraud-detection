
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

url = "sampleData.csv"

creditcard = pd.read_csv(url)
print(creditcard)


print(creditcard.shape)


print(creditcard.columns)

creditcard["TARGET"].value_counts()

sns.set_style("whitegrid")
sns.FacetGrid(creditcard, hue="TARGET", size = 6).map(plt.scatter, "REGION_RATING_CLIENT_W_CITY", "REG_CITY_NOT_WORK_CITY").add_legend()
plt.show()

from scipy import spatial
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sampleData = creditcard.head(20000)  

samples = creditcard.loc[30401:30500] 

frame = []

for i in range(30401, 30501):
    t1 = samples.loc[i]
    c = samples.loc[i]["TARGET"]
    for j in range(20000):
        t2 = sampleData.loc[j]
        classLabel = creditcard.loc[j]["TARGET"]
        similarity = 1 - spatial.distance.cosine(t1, t2)
        frame.append([classLabel, similarity, j])
        
    df = pd.DataFrame(frame, columns=['TARGET', 'Similarity', 'Transaction ID'])
    df_sorted = df.sort_values("Similarity", ascending=False)
    print("Top 10 transactions having highest similarity with transaction ID = "+str(i)+" and target = "+str(c)+":")
    print(df_sorted.iloc[:10])
    print("\n")
    frame = []

