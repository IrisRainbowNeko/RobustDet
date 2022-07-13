import argparse
import yaml

yaml.warnings({'YAMLLoadWarning': False})

class cfgParser:
    def __init__(self, base_block=['model', 'data', 'attack'], sub_block=['']):
        self.parser = argparse.ArgumentParser(description='RobustDet')
        self.parser.add_argument('--cfg', default=None, type=str)
        self.base_block=base_block

    def parse_args(self, cfg_data, remaining):
        for i in range(0, len(remaining), 2):
            k, v = remaining[i:i+2]
            k=k[2:]

            if hasattr(cfg_data, k):
                if getattr(cfg_data, k) is None:
                    setattr(cfg_data, k, eval(v))
                else:
                    t=type(getattr(cfg_data, k))
                    if t in [bool, float]:
                        setattr(cfg_data, k, eval(v))
                    else:
                        setattr(cfg_data, k, t(v))
        return cfg_data

    def load_cfg(self, add_block):
        cfg_path, remaining = self.parser.parse_known_args()
        with open(cfg_path.cfg, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        args=argparse.Namespace()
        for block in self.base_block+add_block:
            for k,v in cfg[block].items():
                setattr(args, k, v)

        args = self.parse_args(args, remaining)
        return args