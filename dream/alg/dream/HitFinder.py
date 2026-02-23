import numpy as np


class HitFinder:

    def __init__(self, params):
                              
        self.uRunTime = params['runtime_u']
        
        self.vRunTime = params['runtime_v']
        
        self.wRunTime = params['runtime_w']
        
        self.u_diff_offset = params['u_diff_offset']
        
        self.v_diff_offset = params['v_diff_offset']
        
        self.w_diff_offset = params['w_diff_offset']        
        
        
        self.uTSumAvg = params['tsum_avg_u']
        self.uTSumLow = self.uTSumAvg - params['tsum_hw_u']
        self.uTSumHigh = self.uTSumAvg + params['tsum_hw_u']
        
        self.vTSumAvg = params['tsum_avg_v'] 
        self.vTSumLow = self.vTSumAvg - params['tsum_hw_v']
        self.vTSumHigh = self.vTSumAvg + params['tsum_hw_v']     
          
        self.wTSumAvg = params['tsum_avg_w']
        self.wTSumLow = self.wTSumAvg - params['tsum_hw_w']
        self.wTSumHigh = self.wTSumAvg + params['tsum_hw_w']
       

        
        self.radius2 = params['rMCP']**2
        self.f_u = params['f_u']
        self.f_v = params['f_v']
        self.f_w = params['f_w']
        
         
        self.sqrt3 = np.sqrt(3.)
        self.reconstruction_k_diag_tsum = False
        self.reconstruction_k_diag_diff = False
        
    
        
        
    def FindHits(self, McpSig, u1Sig, u2Sig, v1Sig, v2Sig, w1Sig, w2Sig):

        t1u = (-self.uRunTime+2*McpSig+self.uTSumAvg-np.abs(self.u_diff_offset))/2
        t2u = (self.uRunTime+2*McpSig+self.uTSumAvg+np.abs(self.u_diff_offset))/2
            
        t1v = (-self.vRunTime+2*McpSig+self.vTSumAvg-np.abs(self.v_diff_offset))/2
        t2v = (self.vRunTime+2*McpSig+self.vTSumAvg+np.abs(self.v_diff_offset))/2
            
        t1w = (-self.wRunTime+2*McpSig+self.wTSumAvg-np.abs(self.w_diff_offset))/2
        t2w = (self.wRunTime+2*McpSig+self.wTSumAvg+np.abs(self.w_diff_offset))/2           
        
        
        self.list_dict = {}

        if self.reconstruction_k_diag_tsum:
            for k in ['u','v','w']:
                self.list_dict['tsum_'+k] = [] 

        if self.reconstruction_k_diag_diff:
            for k in ['u','v','w']:
                self.list_dict['diff_'+k] = []        
            
        self.list_dict['x'] = []
        self.list_dict['y'] = []
        self.list_dict['t'] = []

        self.data_dict = {}

        # if self.reconstruction_k_diag_tsum:
        #     for k in ['u','v','w']:
        #         self.data_dict['tsum_'+k] = np.array([])  

        # if self.reconstruction_k_diag_diff:
        #     for k in ['u','v','w']:
        #         self.data_dict['diff_'+k] = np.array([])  
    
               
        for i_McpT, McpT in enumerate(McpSig):
           
            u1 = u1Sig[(u1Sig>t1u[i_McpT]) & (u1Sig<t2u[i_McpT])]
            u2 = u2Sig[(u2Sig>t1u[i_McpT]) & (u2Sig<t2u[i_McpT])]
            v1 = v1Sig[(v1Sig>t1v[i_McpT]) & (v1Sig<t2v[i_McpT])]
            v2 = v2Sig[(v2Sig>t1v[i_McpT]) & (v2Sig<t2v[i_McpT])]      
            w1 = w1Sig[(w1Sig>t1w[i_McpT]) & (w1Sig<t2w[i_McpT])]
            w2 = w2Sig[(w2Sig>t1w[i_McpT]) & (w2Sig<t2w[i_McpT])]                          
            
            u1u2_sum = u1[:,np.newaxis] + u2[np.newaxis,:] - 2*McpT 
            v1v2_sum = v1[:,np.newaxis] + v2[np.newaxis,:] - 2*McpT 
            w1w2_sum = w1[:,np.newaxis] + w2[np.newaxis,:] - 2*McpT 
            
            
            u1_ind, u2_ind = np.where((u1u2_sum>self.uTSumLow) & (u1u2_sum<self.uTSumHigh))
            v1_ind, v2_ind = np.where((v1v2_sum>self.vTSumLow) & (v1v2_sum<self.vTSumHigh))
            w1_ind, w2_ind = np.where((w1w2_sum>self.wTSumLow) & (w1w2_sum<self.wTSumHigh))
                    
          
            sum_u = u1[u1_ind]+u2[u2_ind] - 2*McpT - self.uTSumAvg
            sum_v = v1[v1_ind]+v2[v2_ind] - 2*McpT - self.vTSumAvg 
            sum_w = w1[w1_ind]+w2[w2_ind] - 2*McpT - self.wTSumAvg             
            
            if self.reconstruction_k_diag_tsum:
                if sum_u.size > 0: self.list_dict['tsum_u'].append(self.pick_one(sum_u))
                if sum_v.size > 0: self.list_dict['tsum_v'].append(self.pick_one(sum_v))
                if sum_w.size > 0: self.list_dict['tsum_w'].append(self.pick_one(sum_w))
                
                # self.data_dict['tsum_u'] = np.concatenate([self.data_dict['tsum_u'], self.pick_one(sum_u)],axis=0)
                # self.data_dict['tsum_v'] = np.concatenate([self.data_dict['tsum_v'], self.pick_one(sum_v)],axis=0)
                # self.data_dict['tsum_w'] = np.concatenate([self.data_dict['tsum_w'], self.pick_one(sum_w)],axis=0)   

            
            sub_u = u1[u1_ind]-u2[u2_ind]
            sub_v = v1[v1_ind]-v2[v2_ind]
            sub_w = w1[w1_ind]-w2[w2_ind]

            if self.reconstruction_k_diag_diff:
                if sub_u.size > 0: self.list_dict['diff_u'].append(self.pick_one(sub_u))
                if sub_v.size > 0: self.list_dict['diff_v'].append(self.pick_one(sub_v))
                if sub_w.size > 0: self.list_dict['diff_w'].append(self.pick_one(sub_w))
                
                # self.data_dict['diff_u'] = np.concatenate([self.data_dict['diff_u'], self.pick_one(sub_u)],axis=0)
                # self.data_dict['diff_v'] = np.concatenate([self.data_dict['diff_v'], self.pick_one(sub_v)],axis=0)
                # self.data_dict['diff_w'] = np.concatenate([self.data_dict['diff_w'], self.pick_one(sub_w)],axis=0)                
            
            sub_uf = (sub_u-self.u_diff_offset)*self.f_u/2
            sub_vf = (sub_v-self.v_diff_offset)*self.f_v/2
            sub_wf = (sub_w-self.w_diff_offset)*self.f_w/2
                            
            Xuv = sub_uf[:,np.newaxis] + 0*sub_vf[np.newaxis,:]
            Yuv = (sub_uf[:,np.newaxis] - 2*sub_vf[np.newaxis,:])/self.sqrt3              
            x = np.ravel(Xuv)
            y = np.ravel(Yuv)

            if len(x) > 0:
                tsum_abs = np.abs(sum_u[:,np.newaxis]) + np.abs(sum_v[np.newaxis,:])
                tsum_abs = np.ravel(tsum_abs)         
                ind_min = np.argmin(tsum_abs)
                x1, y1 = x[ind_min], y[ind_min]
                rMCP2 = x1**2+y1**2              
                if rMCP2<self.radius2:                    
                    self.list_dict['x'].append(x1)
                    self.list_dict['y'].append(y1)
                    self.list_dict['t'].append(McpT)          

        if self.reconstruction_k_diag_tsum: 
            for k in ['tsum_u', 'tsum_v', 'tsum_w']:
                self.data_dict[k] = np.array(self.list_dict[k])

        if self.reconstruction_k_diag_diff: 
            for k in ['diff_u', 'diff_v', 'diff_w']:
                self.data_dict[k] = np.array(self.list_dict[k])        
                           
        for k in ['x', 'y', 't']:
            self.data_dict[k] = np.array(self.list_dict[k])
        
        self.data_dict['n'] = np.array([len(self.data_dict['t'])])


    
    def pick_one(self, arr):
        # if arr.size == 0:
        #     return None   
        return np.random.choice(arr)

    def pick_one_arr(self, arr):
        if arr.size == 0:
            return np.array([])      
        return np.random.choice(arr, size=1, replace=False)



