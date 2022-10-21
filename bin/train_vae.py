import argparse
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/..')

from galaxy_generator.vae_generator import VAE_Generator
from galaxy_generator.utils import get_config_from_yaml


def main(args=None):
    '''
        Usage:
            run a test (on genie)
            >> cd /home/hhg/Research/galaxyClassify/repo/GalaxyZooGenerator/
            >> mkdir experiments
            >> python3 bin/train_vae.py --config configs/vae_test.yaml > experiments/vae_test.log
            >> python3 bin/train_vae.py --config configs/vae_run0.yaml > experiments/vae_run0.log
            >> CUDA_LAUNCH_BLOCKING=1 python3 bin/train_vae.py --config configs/vae_run1.yaml > experiments/vae_run1.log
        
        p.s. To delete the test file
            >> rm -rf ./experiments/vae_test/
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='config file for training settings')

    opt = parser.parse_args()
    print(opt)

    config = get_config_from_yaml(opt.config)
    generator = VAE_Generator(config=config)
    generator.train()


if __name__ == '__main__':
    main()
