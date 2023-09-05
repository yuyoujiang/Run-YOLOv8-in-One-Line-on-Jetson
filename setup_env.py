import os
import subprocess
import re
import sys


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


def prepare_running_env():
    # Prepare running environment (most important)
    # Install Ultralytics Package
    print('Step 1. Access the terminal of Jetson device, install pip and upgrade it')
    print('Upgrade pip...')
    user_name = run_cmd('echo $USER')
    os.environ['PATH'] += os.pathsep + os.path.join('/home', user_name.stdout.decode('gbk').strip(), '.local/bin')
    run_cmd('sudo apt-get update')
    run_cmd('sudo apt-get install -y python3-pip')
    run_cmd('pip3 install --upgrade pip')

    # Check python version
    check_python = run_cmd('python3 --version')
    if '3.8' not in check_python.stdout.decode('gbk'):
        print('Please update the python version to 3.8')
        sys.exit(0)

    # Check nvidia-jetpack version
    check_jetpack = run_cmd('sudo dpkg -l | grep -w jetpack')
    if 'nvidia-jetpack' not in check_jetpack.stdout.decode('gbk'):
        print('Installing nvidia-jetpack...(about 5 minutes to wait)')
        run_cmd('sudo apt install nvidia-jetpack -y')
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
        torch_version = re.findall(pattern, check_torch.stdout.decode('gbk'))
        if len(torch_version) == 2:
            if torch_version[0] in ['2.0.0'] and torch_version[1] in ['23.5']:
                print('The torch version is legal')
                install_flag = False
        else:
            print('Uninstalling torch and torchvision...')
            run_cmd('pip3 uninstall torch torchvision -y')
    else:
        print('No torch installed through pip3')
        run_cmd('pip3 uninstall torch torchvision -y')

    if not install_flag:
        print('Check torchvision ...')
        check_torchvision = run_cmd('pip3 list | grep -w torchvision')
        if not check_torchvision.returncode:
            pattern = r'\d+\.(?:\d+\.)*\d+'
            check_torchvision = re.findall(pattern, check_torchvision.stdout.decode('gbk'))
            if len(check_torchvision) == 1 and check_torchvision[0] in ['0.15.2']:
                print('The torchvision version is legal')
            else:
                print('The torchvision is illegal reinstall torch and torchvision...')
                install_flag = True
                run_cmd('pip3 uninstall torch torchvision -y')
        else:
            print('No torchvision installed through pip3')
            install_flag = True
            run_cmd('pip3 uninstall torch torchvision -y')

    if install_flag:
        print('Installing dependency packages for torch...')
        script_cache_path = os.path.join(sys.path[0], 'cache')
        if not os.path.exists(script_cache_path):
            os.makedirs(script_cache_path)
        run_cmd('sudo apt-get update', _stderr=subprocess.STDOUT)
        run_cmd('sudo apt-get install -y libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev')
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

    print('Updating related dependency packages of TensorRT...')
    run_cmd('pip3 install onnx')
    run_cmd('pip3 install numpy==1.20.3', _stderr=subprocess.STDOUT)
    print('Running environment configuration completed!')


if __name__ == '__main__':
    prepare_running_env()

