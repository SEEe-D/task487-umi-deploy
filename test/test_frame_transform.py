import numpy as np
import scipy.spatial.transform as st

tx_flangerotx = np.identity(4)
tx_flangerotx[:3, :3] = st.Rotation.from_euler('x', [np.pi / 2]).as_matrix()

tx_flangerotz = np.identity(4)
tx_flangerotz[:3, :3] = st.Rotation.from_euler('z', [np.pi / 2]).as_matrix()


tx_flange_tip = tx_flangerotx  @ tx_flangerotz
tx_tip_flange = np.linalg.inv(tx_flange_tip)

print("UR7===tx_flange_tip:", tx_flange_tip)


tx_flange_tip = tx_flangerotz @ tx_flangerotx
tx_tip_flange = np.linalg.inv(tx_flange_tip)

print("UR7===tx_flange_tip111:", tx_flange_tip)
