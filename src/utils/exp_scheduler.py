from easydict import EasyDict as edict
import yaml
class ConfigYAML():
    def __init__(self,file_path) -> None:
        self.file_path = file_path
        with open(file_path, 'r') as stream:
            self.data = yaml.safe_load(stream)
        self.data = edict(self.data)
    def __str__(self) -> str:
        return str(self.data)
    def __call__(self, ):
        return self.data
    
if __name__ == "__main__":
    c = ConfigYAML('configs/pecnet/eth.yaml')
    print(c)
    print(c().gpu_idx)