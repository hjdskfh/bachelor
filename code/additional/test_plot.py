import numpy as np
y = [1, 2, 3]
x = [4, 5, 6]
area = np.trapz(y, x)
print(area)  # Should work!
import numpy as np
print(np.__version__)