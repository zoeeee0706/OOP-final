"""
agent.py - 智慧交通號誌控制 Agent (Volume-Based Control)

注意事項 (Notes for Developers):
1. 這個檔案是用來實作 Agent 的邏輯。
2. 使用者執行 `py main.py agent <size> [vehicle_count]` 時，主程式會呼叫這裡的 `step` 函式。
3. 您可以通過 `env` 參數與環境互動。
"""

import random

# Module-level state to store timing info for each intersection
# Key: intersection_position (tuple), Value: dict of state info
intersection_controllers = {}

# Parameters
MIN_GREEN_TIME = 5   # Minimum green light duration (seconds)
MAX_GREEN_TIME = 30  # Maximum green light duration (seconds)
TIME_PER_CAR = 3     # Seconds added per waiting vehicle
YELLOW_TIME = 2      # Yellow light duration (seconds)
DETECTION_DISTANCE = 3  # How many cells ahead to detect waiting vehicles

def get_waiting_counts(env, intersection_pos):
    """
    Count vehicles waiting at the intersection.
    Now checks up to DETECTION_DISTANCE cells in each direction.
    Returns: (ns_count, ew_count)
    """
    ix, iy = intersection_pos
    ns_count = 0
    ew_count = 0
    
    # Get all vehicles
    vehicles = env.get_vehicle_states()
    
    # Check multiple cells in each direction for waiting cars
    # intersection_pos is (x, y)
    
    for v in vehicles:
        vx, vy = v['position']
        v_dir = v['direction'] # 'north', 'south', 'east', 'west'
        
        # NS Traffic (Vehicles on Y-axis approaching intersection)
        # From North (y-1 to y-DETECTION_DISTANCE) facing South
        for dist in range(1, DETECTION_DISTANCE + 1):
            if vx == ix and vy == iy - dist and v_dir == 'south':
                ns_count += 1
                break
        # From South (y+1 to y+DETECTION_DISTANCE) facing North
        for dist in range(1, DETECTION_DISTANCE + 1):
            if vx == ix and vy == iy + dist and v_dir == 'north':
                ns_count += 1
                break
            
        # EW Traffic (Vehicles on X-axis approaching intersection)
        # From West (x-1 to x-DETECTION_DISTANCE) facing East
        for dist in range(1, DETECTION_DISTANCE + 1):
            if vy == iy and vx == ix - dist and v_dir == 'east':
                ew_count += 1
                break
        # From East (x+1 to x+DETECTION_DISTANCE) facing West
        for dist in range(1, DETECTION_DISTANCE + 1):
            if vy == iy and vx == ix + dist and v_dir == 'west':
                ew_count += 1
                break
            
    return ns_count, ew_count

def calculate_duration(car_count):
    """Calculate green light duration based on car count."""
    duration = MIN_GREEN_TIME + (car_count * TIME_PER_CAR)
    return min(MAX_GREEN_TIME, duration)

def step(env):
    """
    Agent 的決策邏輯
    
    這個函式會在每個模擬步驟被呼叫一次。
    現在包含黃燈過渡階段：ns_green → ns_yellow → ew_green → ew_yellow → ns_green
    """
    global intersection_controllers
    
    # 1. 觀察環境 - 取得所有路口狀態
    intersections = env.get_intersection_states()
    
    for state in intersections:
        pos = state['position']
        
        # Initialize controller state if new
        if pos not in intersection_controllers:
            # Determine current phase from env state
            if state['ns_state'] == 'green':
                current_phase = 'ns_green'
            elif state['ns_state'] == 'yellow':
                current_phase = 'ns_yellow'
            elif state['ew_state'] == 'green':
                current_phase = 'ew_green'
            elif state['ew_state'] == 'yellow':
                current_phase = 'ew_yellow'
            else:
                current_phase = 'ns_green'  # Default
            
            intersection_controllers[pos] = {
                'phase': current_phase,  # 'ns_green', 'ns_yellow', 'ew_green', 'ew_yellow'
                'time_remaining': 0      # Will force immediate calculation
            }
        
        controller = intersection_controllers[pos]
        
        # Decrement timer
        controller['time_remaining'] -= 1
        
        # Check if phase switch is needed (Timer expired)
        if controller['time_remaining'] <= 0:
            # Get car counts to decide next duration
            ns_count, ew_count = get_waiting_counts(env, pos)
            
            # State machine: ns_green → ns_yellow → ew_green → ew_yellow → ns_green
            if controller['phase'] == 'ns_green':
                # Green time expired, switch to yellow
                controller['phase'] = 'ns_yellow'
                controller['time_remaining'] = YELLOW_TIME
            elif controller['phase'] == 'ns_yellow':
                # Yellow time expired, switch to EW green
                controller['phase'] = 'ew_green'
                # Calculate time for EW based on EW traffic
                controller['time_remaining'] = calculate_duration(ew_count)
            elif controller['phase'] == 'ew_green':
                # Green time expired, switch to yellow
                controller['phase'] = 'ew_yellow'
                controller['time_remaining'] = YELLOW_TIME
            elif controller['phase'] == 'ew_yellow':
                # Yellow time expired, switch to NS green
                controller['phase'] = 'ns_green'
                # Calculate time for NS based on NS traffic
                controller['time_remaining'] = calculate_duration(ns_count)
            
            # Debug output (optional)
            # print(f"[Agent] Int {pos}: Switch to {controller['phase']} for {controller['time_remaining']}s (Cars: NS={ns_count}, EW={ew_count})")

        # Apply control based on current phase
        if controller['phase'] == 'ns_green':
            env.control_intersection(pos, 'ns_green')
        elif controller['phase'] == 'ns_yellow':
            env.control_intersection(pos, 'ns_yellow')
        elif controller['phase'] == 'ew_green':
            env.control_intersection(pos, 'ew_green')
        elif controller['phase'] == 'ew_yellow':
            env.control_intersection(pos, 'ew_yellow')

    # 3. 重要：必須手動更新紅綠燈計時器
    env.update_intersections()
