import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

x = [1]
for i in x:
    plt.plot(i, i, marker=(3, 0, i), markersize=20, linestyle='None')

plt.xlim([0,4])
plt.ylim([0,4])

plt.savefig('/Titan/code/robohike_ws/src/litevloc/paper_writing/python/test/test_viz_marker.jpg')