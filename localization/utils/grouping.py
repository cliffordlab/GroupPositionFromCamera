import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def detect_direct_interaction(p1, vec1, p2, vec2, file_fig=None):
  '''
  1. Center at the middle point
  2. Rotate to align with X-axis
  3. Get vector angles
  4. Determine if they are interacting
    - face-to-face case
    - facing to similar direction case
  '''

  if file_fig is not None:
    fig, ax = plt.subplots(ncols=3,figsize=(15,5))

  ref1 = p2 - p1
  ref1_norm = np.sqrt(np.sum(ref1**2))
  if ref1_norm > 0:
    ref1 /= ref1_norm
  ref1_end = p1+ref1*10
  vec1_end = p1+vec1*10

  ref2 = p1 - p2
  ref2_norm = np.sqrt(np.sum(ref2**2))
  if ref2_norm > 0:
    ref2 /= ref2_norm
  ref2_end = p2+ref2*10
  vec2_end = p2+vec2*10

  if file_fig is not None:
    ax[0].plot([-30, 30], [0, 0], 'k')
    ax[0].plot([0, 0], [-30, 30], 'k')

    ax[0].plot(p1[0], p1[1], 'bo')
    ax[0].plot(ref1_end[0], ref1_end[1], 'b^')
    ax[0].plot([p1[0], ref1_end[0]], [p1[1], ref1_end[1]], 'b')

    ax[0].plot(vec1_end[0], vec1_end[1], 'r^')
    ax[0].plot([p1[0], vec1_end[0]], [p1[1], vec1_end[1]], 'r')

    ax[0].plot(p2[0], p2[1], 'go')
    ax[0].plot(ref2_end[0], ref2_end[1], 'g^')
    ax[0].plot([p2[0], ref2_end[0]], [p2[1], ref2_end[1]], 'g')

    ax[0].plot(vec2_end[0], vec2_end[1], 'm^')
    ax[0].plot([p2[0], vec2_end[0]], [p2[1], vec2_end[1]], 'm')

  # 1. center at the middle point
  mid = (p1 + p2)/2.

  p1 -= mid
  ref1_end -= mid
  vec1_end -= mid

  p2 -= mid
  ref2_end -= mid
  vec2_end -= mid

  if file_fig is not None:
    ax[1].plot([-30, 30], [0, 0], 'k')
    ax[1].plot([0, 0], [-30, 30], 'k')

    ax[1].plot(p1[0], p1[1], 'bo')
    ax[1].plot(ref1_end[0], ref1_end[1], 'b^')
    ax[1].plot([p1[0], ref1_end[0]], [p1[1], ref1_end[1]], 'b')

    ax[1].plot(vec1_end[0], vec1_end[1], 'r^')
    ax[1].plot([p1[0], vec1_end[0]], [p1[1], vec1_end[1]], 'r')

    ax[1].plot(p2[0], p2[1], 'go')
    ax[1].plot(ref2_end[0], ref2_end[1], 'g^')
    ax[1].plot([p2[0], ref2_end[0]], [p2[1], ref2_end[1]], 'g')

    ax[1].plot(vec2_end[0], vec2_end[1], 'm^')
    ax[1].plot([p2[0], vec2_end[0]], [p2[1], vec2_end[1]], 'm')

  # 2. rotate to align with X-axis
  ref1 /= np.sqrt(np.sum(ref1**2))
  align_rad = np.arctan2(ref1[1], ref1[0])
  align_deg = np.degrees(align_rad)

  c, s = np.cos(-align_rad), np.sin(-align_rad)
  align_rot = np.array([[c, -s],
                      [s, c]])

  p1 = align_rot @ p1
  ref1_end = align_rot @ ref1_end
  vec1_end = align_rot @ vec1_end

  p2 = align_rot @ p2
  ref2_end = align_rot @ ref2_end
  vec2_end = align_rot @ vec2_end

  if file_fig is not None:
    ax[2].plot([-30, 30], [0, 0], 'k')
    ax[2].plot([0, 0], [-30, 30], 'k')

    ax[2].plot(p1[0], p1[1], 'bo')
    ax[2].plot(ref1_end[0], ref1_end[1], 'b^')
    ax[2].plot([p1[0], ref1_end[0]], [p1[1], ref1_end[1]], 'b')

    ax[2].plot(vec1_end[0], vec1_end[1], 'r^')
    ax[2].plot([p1[0], vec1_end[0]], [p1[1], vec1_end[1]], 'r')

    #--
    ax[2].plot(p2[0], p2[1], 'go')
    ax[2].plot(ref2_end[0], ref2_end[1], 'g^')
    ax[2].plot([p2[0], ref2_end[0]], [p2[1], ref2_end[1]], 'g')

    ax[2].plot(vec2_end[0], vec2_end[1], 'm^')
    ax[2].plot([p2[0], vec2_end[0]], [p2[1], vec2_end[1]], 'm')

  # 3. get vector angles
  ref1 = ref1_end - p1
  ref1_norm = np.sqrt(np.sum(ref1**2))
  if ref1_norm > 0:
    ref1 /= ref1_norm

  vec1 = vec1_end - p1
  vec1_norm = np.sqrt(np.sum(vec1**2))
  if vec1_norm > 0:
    vec1 /= vec1_norm

  theta_2 = np.arctan2(vec1[1], vec1[0])
  angle_result_1 = np.degrees(theta_2)

  ref2 = ref2_end - p2
  ref2_norm = np.sqrt(np.sum(ref2**2))
  if ref2_norm > 0:
    ref2 /= ref2_norm

  vec2 = vec2_end - p2
  vec2_norm = np.sqrt(np.sum(vec2**2))
  if vec2_norm > 0:
    vec2 /= vec2_norm

  theta_1 = np.arctan2(ref2[1], ref2[0])
  theta_2 = np.arctan2(vec2[1], vec2[0])
  angle_1 = np.degrees(theta_1)
  if angle_1 < 0:
    angle_1 += 360
  elif np.abs(angle_1) < 10-6:
    angle_1 = 360
  angle_2 = np.degrees(theta_2)
  if angle_2 < 0:
    angle_2 += 360
  elif np.abs(angle_2) < 10-6:
    angle_2 = 360
  angle_result_2 = angle_2 - angle_1

  vec12 = np.dot(vec1, vec2)
  vec12 = np.clip(vec12, -1.0, 1.0)
  angle_result_3 = np.arccos(vec12)
  angle_result_3 = np.degrees(angle_result_3)
  
  # 4. Determine if they are interacting
    # - face-to-face case
    # - facing to similar direction case
  if (angle_result_1 >= -15 and angle_result_1 <= 90 \
    and angle_result_2 >= -90 and angle_result_2 <= 15) \
    or (angle_result_1 >= -90 and angle_result_1 <= 15 \
    and angle_result_2 >= -15 and angle_result_2 <= 90) \
    or (angle_result_3 <= 30):
    is_interact = True
    # print ('relaxed: interact!')
  else:
    is_interact = False
    # print ('relaxed: not interacting ...')

  if file_fig is not None:
    fig.suptitle(f'p1: {angle_result_1} | p2: {angle_result_2} | fd: {angle_result_3} | interact: {is_interact}')
    
    plt.tight_layout()
    plt.savefig(file_fig)
    print ('save in ...', file_fig)

  return is_interact