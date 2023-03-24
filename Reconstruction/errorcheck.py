import matplotlib.pyplot as plt
import numpy as np
  
x = np.array([1, 2, 5, 10])
y_ap_centers = [4.136540908, 11.84827375, 25.9543949, 42.19108557]
y_lat_centers = [4.420301343, 9.032542483, 25.39717331, 48.85609966]
y_rotation_angles = [7.321644695, 15.29112991, 32.47478362, 40.62813564]
y_world_position = [7.663010164, 15.3594023, 55.99961726, 139.5390898]
  
plt.plot(x, y_world_position)
  

plt.xlabel("Percentage Noise Induced")
plt.ylabel("Error in World Position (units)")
plt.title('Variation of World Position Error with Noise')
  
plt.show()
