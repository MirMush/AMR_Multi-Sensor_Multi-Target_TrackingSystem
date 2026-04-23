import json
import os
import numpy as np
from coord_frame_manager import CoordFrameManager

# 1. Setup the file path
file_path = r"C:\Users\mirmu\OneDrive\Desktop\DTU SEM2\Marine Autonomous Robotics\ProjectWork\harbour_sim_output\scenario_E.json"

# 2. Initialize the manager and load the data
cfm = CoordFrameManager()

with open(file_path, 'r') as f:
    data = json.load(f)

# Print the scenario name directly from the JSON metadata
print(f"\n{'='*60}")
print(f"RUNNING SCENARIO: {data['scenario_name']}")
print(f"{'='*60}\n")

measurements = data['measurements']
gt_data = data['ground_truth']#gets the ground truth data for the vessel (pN, pE, vN, vE) at each timestamp,
vessel_positions = data['vessel_positions']
# Helper to find ground truth for h and H validation
def get_gt_state(t,target_id):
    key=str(target_id)
    if key not in gt_data:
        return None
    rows=gt_data[key]
    times = [row[0] for row in rows] #Finds the index of the closest timestamp to the requested t
    idx = np.argmin(np.abs(np.array(times) - t))
    return np.array(rows[idx][1:]) # [pN, pE, vN, vE]
def get_vessel_pos(t):
    times = [row[0] for row in vessel_positions]
    idx = np.argmin(np.abs(np.array(times) - t))
    return vessel_positions[idx][1], vessel_positions[idx][2]  # pN, pE
# Main Processing Loop
seen_targets = set()
print(f"Starting full processing for {len(measurements)} measurements...\n")
for m in measurements: 
    t = m['time']
    s_id = m['sensor_id']
    target_id = m.get('target_id', "Unknown")
    # Update vessel position for the manager
    pN, pE = get_vessel_pos(t)
    cfm.update_vessel_pos(pN, pE)
    if target_id != -1 and target_id not in seen_targets:
        print(f"--- NEW TARGET DETECTED: ID {target_id} at {t}s via {s_id} ---")
        seen_targets.add(target_id)

    # --- SENSOR: GNSS ---
    #if s_id == 'gnss':
        # Update the vessel position (The anchor for all other sensors)
       # cfm.update_vessel_pos(m['north_m'], m['east_m'])
       # print(f"[{t:>5}s] GNSS UPDATE  | Vessel moved to: N={m['north_m']}, E={m['east_m']}")

    # --- SENSOR: AIS ---
if s_id == 'ais':
        # Convert target absolute N/E to a relative Range/Bearing from our current vessel pos
        z_ais_polar = cfm.ais_ned_to_range_bearing(m['north_m'], m['east_m'])
        print(f"[{t:>5}s] AIS RECEIVED | Target at Map[{m['north_m']}, {m['east_m']}] -> Polar{np.round(z_ais_polar, 2)}")

    # --- SENSORS: RADAR & CAMERA ---
elif s_id in ['radar', 'camera'] and not m['is_false_alarm']:
        x_true = get_gt_state(t,m['target_id']) # Get the ground truth state for the target at this timestamp
        
        # Calculate h(x): What the sensor should see based on Ground Truth
        h_pred = cfm.h(x_true, s_id)
        
        # Calculate H(x): The Jacobian (sensitivity matrix)
        H_mat = cfm.H(x_true, s_id)
        
        print(f"[{t:>5}s] {s_id.upper():<11} | Pred h(x): {np.round(h_pred, 2)} | Actual z: [{m['range_m']}, {m['bearing_rad']}]")
        print(f"        Jacobian H row 1 (Range):   {np.round(H_mat[0], 4)}")
        print(f"        Jacobian H row 2 (Bearing): {np.round(H_mat[1], 4)}")

print(f"\n{'='*60}")
print("SCENARIO PROCESSING COMPLETE")
print(f"{'='*60}")