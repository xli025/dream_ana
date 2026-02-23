from typing import Any
import copy
from psana import DataSource

DetectorReturnMap = dict[str, list[str]]          # prefix -> trailing names
DetectorMap = dict[str, DetectorReturnMap]        # detector -> return map

def check_detectors(
    yaml_a: dict[str, Any],
    yaml_b: dict[str, Any],
) -> tuple[list[str], dict[str, Any], DetectorMap]:
    
    # 1) build detector → { prefix: [trailing…] }
    det_maps: DetectorMap = {
        det: (conf.get("return") if isinstance(conf.get("return"), dict) else {})
        for det, conf in yaml_b.items()
    }

    def split_var(v: str) -> tuple[str, str]:
        return tuple(v.split(':', 1)) if ':' in v else (v, v)

    def as_list(field: Any) -> list[str]:
        if isinstance(field, list):
            return field
        if isinstance(field, str):
            return [field]
        return []

    # 2) deep‐copy A for pruning
    updated = copy.deepcopy(yaml_a)
    missing: list[str] = []

    # 3) drop plots exactly as before
    for pname, pconf in list(updated.get('plots', {}).items()):
        to_check: list[str] = []
        to_check += as_list(pconf.get('var'))
        to_check += list(pconf.get('arange', {}).keys())
        to_check += as_list(pconf.get('scan'))
        to_check += as_list(pconf.get('norm'))
       
        if 'func' in pconf.get('type', '') or 'func' in pconf.get('plot_type', ''):

            for key_func in pconf.keys():
           
                if key_func[:4] == 'func':
                    to_check += as_list(pconf[key_func].get('args1'))

        bad = []
        for vs in to_check:
            pre, post = split_var(vs)
            if not any(post in det_maps[d].get(pre, []) for d in det_maps):
                bad.append(vs)

        if bad:
            missing.extend(bad)
            updated['plots'].pop(pname, None)

    # 4) prune data section, handling both normal modes and the special 'x' mode
    for mode_name, mode in updated.get('data', {}).items():
        if mode_name == 'x':
            # handle data/x/scan, epics, bld, timing
            for dtype in ['scan', 'epics', 'bld', 'timing', 'fzp', 'atm']:
                if dtype in mode:
                    vals = as_list(mode[dtype])
                    new_vals = []
                    for post in vals:
                        if any(post in det_map.get(dtype, []) for det_map in det_maps.values()):
                            new_vals.append(post)
                        else:
                            missing.append(f"{dtype}:{post}")
                    if new_vals:
                        mode[dtype] = new_vals
                    else:
                        mode.pop(dtype, None)
        else:
            # existing code for data/<mode>/<prefix>
            for prefix, item in mode.items():
                # fvar unchanged except pruning
                if isinstance(item, dict) and 'fvar' in item:
                    new_f = []
                    for post in item['fvar']:
                        if any(post in det_maps[d].get(prefix, []) for d in det_maps):
                            new_f.append(post)
                        else:
                            missing.append(f"{prefix}:{post}")
                    item['fvar'] = new_f

                # normalize & prune 'var'
                vals = as_list(item.get('var'))
                new_v = []
                for post in vals:
                    if any(post in det_maps[d].get(prefix, []) for d in det_maps):
                        new_v.append(post)
                    else:
                        missing.append(f"{prefix}:{post}")
                if new_v:
                    item['var'] = new_v
                else:
                    item.pop('var', None)

    if missing:
        print("The following vars are missing:", ", ".join(sorted(set(missing))))

    # 5) build requested_vars_by_detector, now including data/x lists
    requested: DetectorMap = {}
    for det, ret_map in det_maps.items():
        pref_to_trs: DetectorReturnMap = {}

        # from plots (unchanged)
        for pconf in updated.get('plots', {}).values():
            for vs in as_list(pconf.get('var')):
                pre, post = split_var(vs)
                if post in ret_map.get(pre, []):
                    pref_to_trs.setdefault(pre, []).append(post)
            for ak in pconf.get('arange', {}):
                pre, post = split_var(ak)
                if post in ret_map.get(pre, []):
                    pref_to_trs.setdefault(pre, []).append(post)
            for vs in as_list(pconf.get('scan')) + as_list(pconf.get('norm')):
                pre, post = split_var(vs)
                if post in ret_map.get(pre, []):
                    pref_to_trs.setdefault(pre, []).append(post)


            # only if this is a func‐plot, harvest its args1
            if 'func' in pconf.get('type', '') or 'func' in pconf.get('plot_type', ''):
                for key_func, func_conf in pconf.items():
                    if key_func.startswith('func') and isinstance(func_conf, dict):
                        for vs in as_list(func_conf.get('args1', [])):
                            pre, post = split_var(vs)
                            if post in ret_map.get(pre, []):
                                pref_to_trs.setdefault(pre, []).append(post)        

        # from data
        for mode_name, mode in updated.get('data', {}).items():
            if mode_name == 'x':
                # each dtype is a simple list
                for dtype, vals in mode.items():
                    for post in as_list(vals):
                        if post in ret_map.get(dtype, []):
                            pref_to_trs.setdefault(dtype, []).append(post)
            else:
                # original data/<mode>/<prefix> handling
                for prefix, item in mode.items():
                    for post in item.get('fvar', []):
                        if post in ret_map.get(prefix, []):
                            pref_to_trs.setdefault(prefix, []).append(post)
                    for post in as_list(item.get('var')):
                        if post in ret_map.get(prefix, []):
                            pref_to_trs.setdefault(prefix, []).append(post)

        # dedupe
        for pre, lst in pref_to_trs.items():
            seen = set()
            uniq = []
            for x in lst:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            pref_to_trs[pre] = uniq

        if pref_to_trs:
            requested[det] = pref_to_trs

    needed = list(requested.keys())
    return needed, updated, requested


def init(rank, mode, exp, run_num, config, callbacks):
    if mode == 'offline':
        import os, glob
        h5_dir = config['h5']['path1'] + exp + config['h5']['path2']
        h5_path = h5_dir + config['h5']['name1'] + str(run_num) + config['h5']['name2']        
        permissions_mode = 0o775
        os.makedirs(h5_dir, mode=permissions_mode, exist_ok=True)                
        log_dir = config['log']['path1'] + exp + config['log']['path2']   
        os.makedirs(log_dir, mode=permissions_mode, exist_ok=True)   
        
        pattern = h5_path[:-3]+'_*.h5'
        files_to_delete = glob.glob(pattern)+glob.glob(h5_path)    
        if rank==0:
            for file in files_to_delete:
                os.remove(file)
                print(f"Deleted {file}")

        if config.get('live', False):
            os.environ['PS_SMD_MAX_RETRIES'] = str(config.get('wait_time', 60))
        else:
            config['live'] = False

        
        if config['max_events'] is not None:
            ds = DataSource(exp=exp,run=run_num, live = config['live'], max_events=config['max_events'], monitor=False) 
        else:
            ds = DataSource(exp=exp,run=run_num, live = config['live'], monitor=False)             
        
        smd = ds.smalldata(filename=h5_path, batch_size=config['batch_size'])

    elif mode == 'online':
        # ds = DataSource(shmem='tmo_meb1')
        ###
        ds = DataSource(exp='tmo101247125',run=91)             
        #####
        smd = ds.smalldata(batch_size=1, callbacks=callbacks)        
    return ds, smd


