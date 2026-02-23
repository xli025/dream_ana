import numpy as np
from scipy.optimize import bisect

class hsd_peak_finder():
    def __init__(self, det_id, sig_names, mapping, params, requested_vars): 

        self.det_id = det_id
        self.mapping = mapping
        self.params = params
        self.sig_names = sig_names
        self.finder = {}
        for k1, vals in self.params['keys'].items():
            for k2 in vals:
                if 'mcp' in k1:
                    self.finder[self.mapping[k1+k2]] = PyCFD(self.params['mcp'])
                elif 'dream_hsd_lv0' == (k1+k2):
                    self.finder[self.mapping[k1+k2]] = PyCFD(self.params['v1'])   
                elif 'dream_hsd_lw1' == (k1+k2):
                    self.finder[self.mapping[k1+k2]] = PyCFD(self.params['w2'])                      
                else:
                    self.finder[self.mapping[k1+k2]] = PyCFD(self.params['dld'])    
        self.ts_wf = None
        
        self.avail_vars = ['wf', 'pdd', 'tpks', 'hpks', 'len_tpks']
        self.num_keys = len(self.params['keys'].keys())

        self.requested = {}
        self.data_dict = {}
        for k1 in self.avail_vars:
            self.requested[k1] = {}
            k1_p = k1+'_'+self.det_id
            if k1_p in requested_vars:
                self.data_dict[k1_p] = {}
                for k2 in sig_names:
                    if k2 in requested_vars[k1_p]:
                        self.requested[k1][k2] = True
                    else:
                        self.requested[k1][k2] = False
            else:
                for k2 in sig_names:
                    self.requested[k1][k2] = False       
    



    def __call__(self, *args, **kwargs):

        try:
            if self.params['fex']:
                self.find_peaks_fex(*args, **kwargs)  
            else:
                self.find_peaks_raw(*args, **kwargs)  
            if self.num_None == self.num_keys:
                self.peak_exist = False
            else:
                self.peak_exist = True
        except Exception as err:
            self.peak_exist = False

        if not self.peak_exist:
            for k1 in self.avail_vars:
                k1_p = k1+'_'+self.det_id
                self.data_dict[k1_p] = {}
                for k2 in self.sig_names: 
                    if self.requested[k1][k2]:
                        if 'len' in k1_p:
                            self.data_dict[k1_p][k2] = np.array([0])
                        else:
                            self.data_dict[k1_p][k2] = np.array([])    
            

    def find_peaks_fex(self, det, evt):
        self.tpks_dict = {}
        self.len_tpks_dict = {}
        for k in self.data_dict.keys(): self.data_dict[k] = {}
        self.num_None = 0
        
        for k1 in self.params['keys'].keys():
            peaks = det[k1].raw.peaks(evt)  
            wfs = det[k1].raw.waveforms(evt) 
            padded = det[k1].raw.padded(evt) 
            fex_status_2 = det[k1].raw.fex_status(evt)
            if peaks is None:
                #print(k1+' FEX is empty!!!')
                self.num_None += 1
                continue
            
            for i, k2 in enumerate(peaks.keys()):
                fex_status = fex_status_2[k2][0][0][0]
                key_pks = self.mapping[k1+str(k2)]
                starts = np.array(peaks[k2][0][0]).astype('float')
                amps = peaks[k2][0][1]
                tpks_all = np.empty((0,), dtype=float)
                if self.requested['hpks'][key_pks]: hpks_all = np.empty((0,), dtype=float)
                for j, (start, amp) in enumerate(zip(starts, amps)):                    
                    if fex_status>0:
                        #print('FEX wrapped, unwrapping it now.')
                        amp = np.unwrap(amp, period=32768)
                    amp = amp.astype('float')
                    ts = (start + np.arange(len(amp)))*0.1682692307692308
                    
                    tpks = self.finder[key_pks](amp, ts)   
                     
                    if len(tpks)==0: continue                  

                    tpks_all = np.concatenate([tpks_all, tpks])

                    if self.requested['hpks'][key_pks]:
                        hpks = self.finder[key_pks].get_heights(amp, ts, tpks)  
                        hpks_all = np.concatenate([hpks_all, hpks])

                
                self.tpks_dict[key_pks] = tpks_all
                self.len_tpks_dict[key_pks] = np.array([len(tpks_all)])
              
                if self.requested['pdd'][key_pks]:
          
                    if padded is not None: 
                        self.data_dict['pdd_'+self.det_id].update({key_pks: padded[k2][0].astype(float)})    

                if self.requested['wf'][key_pks]:
                    if wfs is not None: 
                        self.data_dict['wf_'+self.det_id].update({key_pks: wfs[k2][0].astype(float)})   

                if self.requested['tpks'][key_pks]:
                    self.data_dict['tpks_'+self.det_id].update({key_pks: self.tpks_dict[key_pks]})

                if self.requested['len_tpks'][key_pks]:
                    self.data_dict['len_tpks_'+self.det_id].update({key_pks: self.len_tpks_dict[key_pks]})
                    
                if self.requested['hpks'][key_pks]:
                    self.data_dict['hpks_'+self.det_id].update({key_pks: hpks_all})

    
        
    def find_peaks_raw(self, det, evt):
        self.tpks_dict = {}
        self.len_tpks_dict = {}   
        for k in self.data_dict.keys(): self.data_dict[k] = {}
        for k1 in self.params['keys'].keys():
            wfs = det[k1].raw.waveforms(evt)      
            for i, k2 in enumerate(wfs.keys()):
                key_pks = self.mapping[k1+str(k2)]
                if self.ts_wf is None:
                   self.ts_wf =  wfs[k2]['times']*1e9   

                if self.requested['wf'][key_pks]:
                    self.data_dict['wf_'+self.det_id].update({key_pks: wfs[k2][0].astype(float)})

                tpks_all = self.finder[key_pks](wfs[k2][0].astype(float), self.ts_wf)
            
                
                self.tpks_dict[key_pks] = tpks_all
                self.len_tpks_dict[key_pks] = np.array([len(tpks_all)])   
                
                if self.requested['tpks'][key_pks]:
                    self.data_dict['tpks_'+self.det_id].update({key_pks: self.tpks_dict[key_pks]})

                if self.requested['len_tpks'][key_pks]:
                    self.data_dict['len_tpks_'+self.det_id].update({key_pks: self.len_tpks_dict[key_pks]})



class PyCFD:

    def __init__(self, params):
        self.sample_interval = params['sample_interval']
        self.delay = int(params['delay']/self.sample_interval)
        self.fraction = params['fraction']
        self.threshold = params['threshold']
        self.walk = params['walk']
        self.polarity = 1 if params['polarity']=='Positive' else -1
        self.timerange_low = params['timerange_low']
        self.timerange_high = params['timerange_high']
        self.offset = params['offset']
        self.xtol = 0.1*self.sample_interval


    def __call__(self, *args, **kwargs):
        return self.find_peaks(*args, **kwargs)    

        
    def NewtonPolynomial3(self,x,x_arr,y_arr):
    
        d_0_1 = (y_arr[1] - y_arr[0])/(x_arr[1] - x_arr[0])
        d_1_2 = (y_arr[2] - y_arr[1])/(x_arr[2] - x_arr[1])
        d_2_3 = (y_arr[3] - y_arr[2])/(x_arr[3] - x_arr[2])
        
        d_0_1_2 = (d_1_2 - d_0_1)/(x_arr[2] - x_arr[0])
        d_1_2_3 = (d_2_3 - d_1_2)/(x_arr[3] - x_arr[1])        
        d_0_1_2_3 = (d_1_2_3 - d_0_1_2)/(x_arr[3] - x_arr[0])
        
        c0 = y_arr[0]
        c1 = d_0_1
        c2 = d_0_1_2
        c3 = d_0_1_2_3
        
        return c0 + c1*(x-x_arr[0]) + c2*(x-x_arr[0])*(x-x_arr[1]) + c3*(x-x_arr[0])*(x-x_arr[1])*(x-x_arr[2])
        
            
    def find_peaks(self,wf, wt):        
        wt_inds = (wt>self.timerange_low)&(wt<self.timerange_high)
        wf = wf[wt_inds] 
        wt = wt[wt_inds] #choose the time window of interest        
        
        wf_1 = wf[:-self.delay] #original waveform
        wf_2 = wf[self.delay:] #delayed waveform
       
        wf_cal = wf_1 - self.fraction*wf_2 #bipolar waveform
        wf_cal_m_walk = self.polarity*wf_cal-self.walk+self.polarity*(self.fraction*self.offset-self.offset) #bipolar signal minus the walk level
        wf_cal_m_walk_sign = np.sign(wf_cal_m_walk) 

        wf_cal_ind = np.where((wf_cal_m_walk_sign[:-1] < wf_cal_m_walk_sign[1:]) & 
        (wf_cal_m_walk_sign[1:] != 0) & ((wf_cal_m_walk[1:] - wf_cal_m_walk[:-1]) >= 1e-8))[0] #find the sign change locations of wf_cal_m_walk

        #check if the orignal signal is above the threhold at sign change locations of wf_cal_m_walk
        wf_cal_ind_ind = np.where(self.polarity*wf_1[wf_cal_ind] > (self.threshold+self.polarity*self.offset))[0]  

        
        t_cfd_arr = np.empty([0,])
        
        
        #The arrival time t_cfd is obtained from the Newton Polynomial fitted to the 4 data points around the location found from above.
        try:
            for ind in wf_cal_ind_ind:

                t_arr = wt[(wf_cal_ind[ind]-1):(wf_cal_ind[ind]+3)]

                wf_cal_m_walk_arr = wf_cal_m_walk[(wf_cal_ind[ind]-1):(wf_cal_ind[ind]+3)]
            
                if len(t_arr) != 4 or len(wf_cal_m_walk_arr) != 4:
                    continue
            
                if (t_arr[1] - t_arr[0])==0 or (t_arr[2] - t_arr[1])==0 or (t_arr[3] - t_arr[2])==0:
                    continue
                
                if (t_arr[2] - t_arr[0])==0 or (t_arr[3] - t_arr[1])==0 or (t_arr[3] - t_arr[0])==0:
                    continue
            
                t_cfd = bisect(self.NewtonPolynomial3,t_arr[1],t_arr[2],args=(t_arr, wf_cal_m_walk_arr),xtol=self.xtol)
            
                t_cfd_arr = np.append(t_cfd_arr,t_cfd)
        except:
            t_cfd_arr = np.append(t_cfd_arr,wt[wf_cal_ind[ind]])

        return t_cfd_arr


    def get_heights(self, wf, wt, t_arr):
        
        k = len(t_arr)
        if k == 0:
            return np.array([], dtype=float)

        wt_inds = (wt>self.timerange_low)&(wt<self.timerange_high)
        #print('wf max 0 :', np.max(wf), 'wf min 0 :', np.min(wf), 'self.offset:', self.offset, 'self.polarity:', self.polarity)
        wf = (wf[wt_inds] - self.offset)*self.polarity 
        wt = wt[wt_inds] #choose the time window of interest         
        
        max_values = np.empty(k, dtype=float)
    
        # indices right after each boundary (right-closed)
        starts = np.searchsorted(wt, t_arr, side='right')
    
        for s in range(k):
            right_boundary = t_arr[s + 1] if s + 1 < k else wt[-1]
            end = np.searchsorted(wt, right_boundary, side='right')
    
            start = starts[s]
            if start < end:
                max_values[s] = np.max(wf[start:end])
            else:
                max_values[s] = np.nan
        # print('wf max:', np.max(wf))
        # print('max_values:',max_values)
        return max_values
