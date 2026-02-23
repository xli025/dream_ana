import time
import os, importlib
from dream.util.setup import check_detectors, init
from dream.util.misc import read_config, read_args, deep_merge
from dream.util.comm import comm_online, comm_offline
from dream.alg.common.x import scan, bld, epics, timing
from dream.util.callback import callback_online

rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))
numworkers = max(size - 1, 1)

mode, exp, run_num = read_args()
if rank==0: print('running '+mode+'...')
if mode == 'online':
    os.environ['PS_SRV_NODES']='1' 
elif numworkers>120 and mode == 'offline':
    SRV_NODES = int(2*numworkers/120)
    EB_NODES = int(3*numworkers/120)
    os.environ['PS_SRV_NODES']=str(SRV_NODES)
    os.environ['PS_EB_NODES']=str(EB_NODES)
else:
    os.environ['PS_SRV_NODES']='1'
    os.environ['PS_EB_NODES']='1' 
    
config_dir = os.getenv("CONFIGDIR")
instrument = read_config(config_dir+'instrument.yaml')['instrument']
config = read_config(config_dir+instrument+'/'+mode+'.yaml') 

config_det = read_config(config_dir+instrument+'/det.yaml')
detectors, config, requested_vars_by_detector = check_detectors(config, config_det)

if rank==0: 
    print(requested_vars_by_detector)

algs = {}
for det in detectors:
    mod = importlib.import_module(config_det[det]['module'])
    alg = getattr(mod, config_det[det]['alg'])
    algs[det] = alg(**config_det[det]['kwargs'], requested_vars = requested_vars_by_detector[det], rank = rank) if 'kwargs' in config_det[det].keys() else alg(requested_vars = requested_vars_by_detector[det])

if mode=='online':
    comm = comm_online(config, requested_vars_by_detector)
    callback = callback_online(rank, numworkers, config)
    callbacks=[callback.smalldata]
else:
    comm = comm_offline(config)
    callbacks = []

while 1: 
    ds, smd = init(rank, mode, exp, run_num, config, callbacks=callbacks) 

    for run in ds.runs():
        dets = {}
        detectors_rm = []
        for det in detectors:
            dets[det] = {}
            if det in ['scan', 'bld', 'epics', 'timing']:
                algs[det].get_det_keys(run)

            try:
                for det_key in algs[det].params['det']['keys']:
                    dets[det][det_key] = run.Detector(det_key) if det_key else det_key     
            except Exception as err:
                dets[det][det_key] = None
                print(err)
                if det not in ['scan', 'bld', 'epics', 'timing']:
                    detectors_rm.append(det)                               
                    if mode=='offline':
                        for var_k in requested_vars_by_detector[det].keys():
                            if 'uniform' in config['data'].keys():
                                if var_k in config['data']['uniform'].keys(): del config['data']['uniform'][var_k]
                            if 'ragged' in config['data'].keys():
                                if var_k in config['data']['ragged'].keys(): del config['data']['ragged'][var_k]
                            if 'ragged_split' in config['data'].keys():
                                if var_k in config['data']['ragged_split'].keys(): del config['data']['ragged_split'][var_k]
                                    
                    del requested_vars_by_detector[det]
                    

        for det in detectors_rm: detectors.remove(det)
            
        priority = {'timing': 0, 'bld': 1}      
        detectors.sort(key=lambda x: priority.get(x, 2))
        
        n_evt = 0
        for step_i, step in enumerate(run.steps()):
            for nevt,evt in enumerate(step.events()):
                
                try:
                    evt_dict = {}     
                    deep_merge(evt_dict, {'x':{'timestamp': evt.timestamp}})
                    for det in detectors:
                        deep_merge(evt_dict, algs[det](dets[det], evt, evt_dict['x']))
                            
                    comm.send(rank, smd, n_evt, evt, evt_dict)
                    n_evt += 1
                
                except Exception as err:
                   print(err)
            
        if mode == 'online': 
            #pass
            smd.event(evt,{'endrun':1}) # tells gatherer to reset plots
        else:
            smd.done() 
        
    if mode == 'offline': break





