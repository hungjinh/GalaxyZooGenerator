import argparse
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/..')

from galaxy_generator.dcgan_generator import DCGAN_Generator
from galaxy_generator.utils import get_config_from_yaml


def main(args=None):
    '''
        Usage:
            run a test (on genie)
            >> cd /home/hhg/Research/galaxyClassify/repo/GalaxyZooGenerator/
            >> mkdir experiments
            >> python3 bin/train_dcgan.py --config configs/dcgan_test.yaml > experiments/test.log
            >> python3 bin/train_dcgan.py --config configs/dcgan_run0.yaml > experiments/run0.log
            >> python3 bin/train_dcgan.py --config configs/dcgan_run1.yaml > experiments/run1.log
            >> python3 bin/train_dcgan.py --config configs/dcgan_run2.yaml > experiments/run2.log
        
        p.s. To delete the test file
            >> rm -rf ./experiments/TestDCGAN/
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='config file for training settings')

    opt = parser.parse_args()
    print(opt)

    config = get_config_from_yaml(opt.config)
    generator = DCGAN_Generator(config=config)
    generator.train()


if __name__ == '__main__':
    main()
