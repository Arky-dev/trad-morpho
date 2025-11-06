import os

# with open(r'C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet\train.en', 'r', encoding='utf-8') as f, open(r'C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet\train.en.sample', 'w+', encoding='utf-8') as g:
#     for i in range(10000):
#         g.write(f.readline().strip()+'\n')
import matplotlib.pyplot as plt

L = [6.2314,
5.1835,
4.5237,
3.9157,
3.3103,
2.7047,
2.1848,
1.7717,
1.4302,
1.1647]

plt.plot(list(range(len(L))), L)
plt.show()