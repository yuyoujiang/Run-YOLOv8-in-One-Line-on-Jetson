import os
import subprocess
import re
import argparse
import sys


model_download_link = {
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',

    'yolov8n-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt',
    'yolov8s-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt',
    'yolov8m-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt',
    'yolov8l-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt',
    'yolov8x-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt',

    'yolov8n-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt',
    'yolov8s-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt',
    'yolov8m-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt',
    'yolov8l-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt',
    'yolov8x-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt',

    'yolov8n-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt',
    'yolov8s-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt',
    'yolov8m-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt',
    'yolov8l-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt',
    'yolov8x-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt',
    'yolov8x-pose-p6': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt'
}


def run_cmd(cmd, _password=None, _stderr=None):
    if _password is None:
        rst = subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=_stderr)
        # print(rst.stdout.decode('gbk'))
    else:
        p1 = subprocess.run('echo ' + _password, stdin=None, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
        rst = subprocess.run(
            cmd, input=p1.stdout, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        # print(rst.stdout)
    return rst


def run_cmd_with_popen(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        try:
            line = p.stdout.readline().decode('gbk')
        except:
            continue
        line = line.strip()
        if line and 'error' not in line:
            print('Popen output: [{}]'.format(line))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('password', type=str, help='administrator password')
    parser.add_argument(
        '--task', default='detect', type=str,
        help='supported "detect", "classify", "segment" and "pose". '
             'Refer to https://docs.ultralytics.com/tasks/ for more information'
    )
    parser.add_argument(
        '--model', default='yolov8n', type=str,
        help='model name. Refer to https://docs.ultralytics.com/models/yolov8/#supported-modes'
    )
    parser.add_argument('--use_trt', default=True, type=bool, help='export engine model')
    parser.add_argument('--use_half', default=True, type=bool, help='FP16 quantization')
    parser.add_argument(
        '--source', default="0", type=str,
        help='path to input video or camera id. Refer to https://docs.ultralytics.com/modes/'
             'predict/#inference-sources for more detailed information.'
    )
    _args = parser.parse_args()
    return _args


def check_args(_args):
    assert _args.model in model_download_link, \
        f'Please enter the correct model name. {model_download_link.keys()}'
    if _args.task not in ["detect", "classify", "segment", "pose"]:
        print(f'Task {_args.task} not support')
        sys.exit(0)
    if _args.task == 'classify':
        assert 'cls' in _args.model, 'Task and model do not match'
    elif _args.task == 'segment':
        assert 'seg' in _args.model, 'Task and model do not match'
    elif _args.task == 'pose':
        assert 'pose' in _args.model, 'Task and model do not match'


def prepare_running_env(cfg):
    # Prepare running environment (most important)
    # Install Ultralytics Package
    print('Step 1. Access the terminal of Jetson device, install pip and upgrade it')
    print('Upgrade pip...')
    os.environ['PATH'] += os.pathsep + '~/.local/bin'
    run_cmd('sudo -S apt-get update', cfg.password)
    run_cmd('sudo -S apt-get install -y python3-pip', cfg.password)
    run_cmd('pip3 install --upgrade pip')

    # user_name = run_cmd('echo $USER')
    # file_path = os.path.join('/home', user_name.stdout.decode('gbk').strip(), '.bashrc')
    # add_path_flag = False
    # with open(file_path, 'r') as f:
    #     lines = f.readlines()
    #     if 'export PATH=~/.local/bin:$PATH\n' not in lines:
    #         add_path_flag = True
    # if add_path_flag:
    #     with open(file_path, 'a') as f:
    #         f.write('\n')
    #         f.write('# For run yolov8 in one line\n')
    #         f.write('export PATH=~/.local/bin:$PATH\n')
    #     run_cmd('sudo -S source ~/.bashrc', cfg.password)

    # Check python version
    check_python = run_cmd('python3 --version')
    if '3.8' not in check_python.stdout.decode('gbk'):
        print('Please update the python version to 3.8')
        sys.exit(0)

    # Check nvidia-jetpack version
    check_jetpack = run_cmd('sudo -S dpkg -l | grep -w jetpack', cfg.password)
    if 'nvidia-jetpack' not in check_jetpack.stdout:
        print('Installing nvidia-jetpack...(about 5 minutes to wait)')
        run_cmd('sudo -S apt install nvidia-jetpack -y', cfg.password)
    else:
        print('nvidia-jetpack is already installed!')

    print('Step 2. Install Ultralytics package')
    check_ultralytics = run_cmd('pip3 list | grep -w ultralytics')
    if check_ultralytics.stdout.decode('gbk') == '':
        print('Installing Ultralytics...')
        run_cmd('pip3 install ultralytics==8.0.150', _stderr=subprocess.STDOUT)
    else:
        print('Ultralytics has been installed')

    # Reinstall Torch and Torchvision
    print('Step 3. Reinstall Torch and Torchvision')
    print('Get JetPack version...')
    check_jetpack_version = run_cmd('cat /etc/nv_tegra_release')
    pattern_jetpack_version1 = r'R\d\d'
    pattern_jetpack_version2 = r'\d+\.(?:\d+\.)*\d+'
    jetpack_version1 = re.findall(pattern_jetpack_version1, check_jetpack_version.stdout.decode('gbk'))
    jetpack_version2 = re.findall(pattern_jetpack_version2, check_jetpack_version.stdout.decode('gbk'))
    jetpack_version = jetpack_version1[0] + '.' + jetpack_version2[0]
    print(f"JetPack: {jetpack_version}")

    install_flag = True
    print('Check torch ...')
    check_torch = run_cmd('pip3 list | grep -w torch')
    if not check_torch.returncode:
        pattern = r'\d+\.(?:\d+\.)*\d+'
        torch_vision = re.findall(pattern, check_torch.stdout.decode('gbk'))
        if len(torch_vision) == 2:
            if torch_vision[0] in ['2.0.0'] and torch_vision[1] in ['23.5']:
                print('The torch version is legal')
                install_flag = False
        else:
            print('Uninstalling torch and torchvision...')
            _ = run_cmd('pip3 uninstall torch torchvision -y')
    else:
        print('No torch installed through pip3')

    if install_flag:
        print('Installing dependency packages for torch...')
        script_cache_path = os.path.join(sys.path[0], 'cache')
        if not os.path.exists(script_cache_path):
            os.makedirs(script_cache_path)
        run_cmd('sudo -S apt-get update', cfg.password, _stderr=subprocess.STDOUT)
        run_cmd('sudo apt-get install -y libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev', cfg.password)
        print('Done!')
        if jetpack_version in ['R35.2.1', 'R35.3.1']:
            torch_package_path = os.path.join(script_cache_path, 'torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl')
            if not os.path.exists(torch_package_path):
                print('Downloading torch package...')
                run_cmd(f'wget https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl'
                        f' -O {torch_package_path}')
            print('Installing torch...')
            run_cmd(f'pip3 install {torch_package_path}', _stderr=subprocess.STDOUT)

            torchvision_package_path = os.path.join(script_cache_path, 'torchvision')
            if not os.path.exists(torchvision_package_path):
                print('Downloading torchvision...')
                run_cmd(f'git clone https://github.com/pytorch/vision {torchvision_package_path}')
            print('Installing torchvision...(about 10 minutes to wait)')
            run_cmd(
                f'cd {torchvision_package_path} ; '
                f'git checkout v0.15.2 ; '
                f'python3 setup.py install --user ; '
                f'cd {os.getcwd()}',
                _stderr=subprocess.STDOUT)
        # elif jetpack_version in ['R34.1', 'R35.1', 'R35.2.1', 'R35.3.1']:
        else:
            torch_package_path = os.path.join(script_cache_path, 'torch-1.13.0+nv22.10-cp38-cp38-linux_aarch64.whl')
            if not os.path.exists(torch_package_path):
                print('Downloading torch package...')
                run_cmd(f'wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0'
                        f'+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl'
                        f' -O {torch_package_path}')
            print('Installing torch...')
            run_cmd(f'pip3 install {torch_package_path}', _stderr=subprocess.STDOUT)

            torchvision_package_path = os.path.join(script_cache_path, 'torchvision')
            if not os.path.exists(torchvision_package_path):
                print('Downloading torchvision...')
                run_cmd(f'git clone https://github.com/pytorch/vision {torchvision_package_path}')
            print('Installing torchvision...(about 10 minutes to wait)')
            run_cmd(
                f'cd {torchvision_package_path} ; '
                f'git checkout v0.14.1 ; '
                f'python3 setup.py install --user ; '
                f'cd {os.getcwd()}',
                _stderr=subprocess.STDOUT)

        print('Reinstall of torch torchvision completed!')

    if cfg.use_trt:
        print('Updating related dependency packages of TensorRT...')
        run_cmd('pip3 install onnx')
        run_cmd('pip3 install numpy==1.20.3', _stderr=subprocess.STDOUT)
    print('Running environment configuration completed!')


if __name__ == '__main__':
    args = parse_args()
    check_args(args)
    prepare_running_env(args)

    model_path = os.path.join(sys.path[0], 'weights', args.task)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, args.model + '.pt')
    if not os.path.exists(model_path):
        print('Downloading model...')
        run_cmd(f"wget {model_download_link[args.model]} -O {model_path}")

    if args.use_trt:
        if not os.path.exists(os.path.splitext(model_path)[0] + '.engine'):
            print('Export model to tensorRT...')
            cmd_str = f"yolo export model={model_path} format=engine device=0"
            if args.use_half:
                cmd_str += ' half=True'
            print('This process may take up to 20 minutes, please be patient and wait...')
            run_cmd(cmd_str, _stderr=subprocess.STDOUT)
            # run_cmd_with_popen(cmd_str)
        model_path = os.path.splitext(model_path)[0] + '.engine'

    run_cmd(f"yolo {args.task} predict model='{model_path}' source='{args.source}' show=True save=False")
