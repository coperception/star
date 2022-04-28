import yaml
import argparse

def parse_config(config_file, mode, ckpt=None):
    with open(config_file,"r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # print(config)
        if mode == 'eval':
            res = dict(config['train'], **config['eval'])
            if ckpt is not None:
                res.update({'resume' : ckpt})
        else: 
            res = config[mode]
            
        res.update({'config_file' : config_file})
        res.update({'data': config['data']+'/'+ config[mode]['split']})
            
        return argparse.Namespace(**res)