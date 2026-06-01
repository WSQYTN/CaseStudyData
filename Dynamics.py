import numpy as np
from typing import List, Dict, Tuple, Optional
from Env.Vehicle import Vehicle
class Dynamics:
    def __init__(self, n_stations: int, Q_p: int, tau_ij: np.ndarray, l_max: int, T: int,d_ijt :np.ndarray):
        self.n_stations = n_stations
        self.Q_p = Q_p
        self.tau_ij = tau_ij
        self.l_max = l_max
        self.T = T
        self.last_arrival_time = T - tau_ij[0, n_stations - 1]
        self.d_ijt = d_ijt
    def fixed_demand(self, a_ijt):
        """
        Converts cumulative d_ijt into arrivals a_ijt[i,j,t] = delta over time.
        """
        I, _, T = self.d_ijt.shape
        for i in range(I):
            for j in range(I):
                if i < j:
                    a_ijt[i,j,0] = self.d_ijt[i,j,0]
                    for t in range(1,T):
                        a_ijt[i,j,t] = max(0, self.d_ijt[i,j,t] - self.d_ijt[i,j,t-1])
        
    def update_waiting_passengers(self, W_ijtt: np.ndarray, a_ijt: np.ndarray, t: int):
        """Update waiting matrix with arrivals from arrivals matrix"""
        #Copy waiting state from previous timestep
        if t <= 0 or t >= self.T:
            return
        W_ijtt[:, :, :, t] = W_ijtt[:, :, :, t-1]
        
        #Add new arrivals from arrivals matrix
        for i in range(self.n_stations-1):
            for j in range(i + 1, self.n_stations):
                if a_ijt[i, j, t] > 0:
                    W_ijtt[i, j, t, t] += a_ijt[i, j, t]
    
    def alighting(self, vehicles: List[Vehicle], A_jt: np.ndarray, t: int):
        """Handle passenger alighting process"""
        if t >= self.T:
            return
        for v in vehicles:
            if v.pos[0] == "station" and v.delta_t == 0 and v.active:
                #print(f"[ALG]  veh={v.id} arriving at station X_j_before={v.X_j.copy()}")
                station = v.pos[1]
                if station >= 1: 
                    A_jt[station,t] += v.X_j[station]
                    v.X_j[station] = 0
                #print(f"[ALG]  veh={v.id} X_j_after={v.X_j.copy()}")
        
    
    def boarding(self, vehicles: List[Vehicle], W_ijtt: np.ndarray, t: int, b_ijtt: np.ndarray):
        """Handle passenger boarding """
        if t >= self.T:
            return
        for v in vehicles:
            
            if (not v.active) or v.pos[0] != "station": #only for active vehicles that are at a station at the current timestep
                continue
            #print(f"[BRD START]  veh={v.id} pos={v.pos} n_p={v.n_p} cap={v.n_p*self.Q_p} Xj={v.X_j.copy()}")
            i = v.pos[1]
            Res = v.get_residual_capacity(self.Q_p) 
            if Res == 0: #no capcaity left 
                continue
            t_prime = 0
            while Res > 0 and t_prime <= t:
                total_fifo = sum(W_ijtt[i, j, t_prime, t] for j in range(i + 1, self.n_stations)) # Total demand at time t' 
                if total_fifo == 0:
                    t_prime += 1
                    continue
                if Res >= total_fifo: #can board the entire group arrive at t' waiting
                    for j in range(i + 1, self.n_stations):
                        w = W_ijtt[i, j, t_prime, t]
                        if w > 0:
                            b_ijtt[i, j, t_prime, t] = w
                            v.X_j[j] += w
                            W_ijtt[i, j, t_prime, t] = 0
                            Res -= w
                else: #Partial proportional boarding  not enough capacity to get the entire group
                    
                    
                    assigned = 0
                    alloc = [] #list of how many passengers can board each destination
                    for j in range(i + 1, self.n_stations):#assign proportions
                        w = W_ijtt[i, j, t_prime, t]
                        if w == 0:
                            alloc.append(0)
                            continue
                        x = (Res * w) / total_fifo
                        x_int = int(x)     # floor
                        x_int = min(x_int, w)  # safety
                        alloc.append(x_int)
                        assigned += x_int
                    remainder = Res - assigned #remainder seats after allocation 
                    if remainder > 0: # keep the fractional part of the proportions to decide where to put the extra seats 
                        fracs = []
                        for j in range(i + 1, self.n_stations):
                            w = W_ijtt[i, j, t_prime, t]
                            if w == 0:
                                fracs.append(-1)
                                continue
                            x = (Res * w) / total_fifo
                            fracs.append(x - int(x))
                        for _ in range(remainder):
                            if all(f <= 0 for f in fracs):
                                break
                            j_rel = max(range(len(fracs)), key=lambda k: fracs[k]) #find index with largest franction
                            if fracs[j_rel] <= 0:
                                break
                            j = i + 1 + j_rel #map it to station index
                            if W_ijtt[i, j, t_prime, t] > alloc[j_rel]: #still waiting passegners after allocation add one seat to the largest farction index
                                alloc[j_rel] += 1
                                fracs[j_rel] = 0  # used up
                            else:
                                fracs[j_rel] = 0  # can't assign here
                    for idx, x_int in enumerate(alloc): # final allocation of the seats
                        j = i + 1 + idx
                        if x_int > 0:
                            x_int = min(x_int, W_ijtt[i, j, t_prime, t])  # safety
                            b_ijtt[i, j, t_prime, t] = x_int
                            v.X_j[j] += x_int
                            W_ijtt[i, j, t_prime, t] -= x_int
                            Res -= x_int
        
                t_prime += 1
            #print(f"[BRD END]    veh={v.id} new Xj={v.X_j.copy()} cap={v.n_p*self.Q_p}")
            #if np.sum(v.X_j) > v.n_p * self.Q_p:
                #print(f"[BRD ERROR] OVERLOAD happened inside boarding! veh={v.id}")
    def update_vehicle_movements(self, vehicles: List[Vehicle]):
        """Update vehicle positions and travel times"""
        for v in vehicles:
            #print(f"[MOVE] BEFORE  veh={v.id} pos={v.pos} dt={v.delta_t}")
            if not v.active:
                continue
            if v.pos[0] == "station" and v.delta_t == 0:# normal forward movement after visitng station
                    i = v.pos[1]
                    if i < self.n_stations - 1:
            
                        dt = self.tau_ij[i, i+1]
                        v.update_pos(("segment", i, i+1), dt)
                    else:# LAST STATION → deactivate vehicle
                        v.active = False
            elif v.pos[0] == "segment": #reaching station
                if v.delta_t == 1:
                    v.update_pos(("station", v.pos[2]), 0)
                elif v.delta_t > 1:
                    v.delta_t -= 1
            #print(f"[MOVE] AFTER   veh={v.id} pos={v.pos} dt={v.delta_t}")
        return vehicles